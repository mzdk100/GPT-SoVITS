mod error;
mod logits_sampler;
mod onnx_builder;
mod text;

use {
    async_stream::stream,
    log::{debug, info},
    ndarray::{
        Array, Array2, ArrayBase, ArrayD, ArrayView2, Axis, IxDyn, OwnedRepr, concatenate, s,
    },
    ort::{
        inputs,
        session::{RunOptions, Session},
        value::{Tensor, TensorRef},
    },
    rodio::{Source, buffer::SamplesBuffer, decoder::Decoder, source::UniformSourceIterator},
    std::{io::Cursor, path::Path, time::SystemTime},
    tokio::fs::read,
};
pub use {
    error::*,
    futures::{Stream, StreamExt},
    logits_sampler::*,
    onnx_builder::*,
    text::*,
};

const T2S_DECODER_EOS: i64 = 1024;
const VOCAB_SIZE: usize = 1025;
const NUM_LAYERS: usize = 24;

type KvDType = f32;

#[derive(Clone)]
pub struct ReferenceData {
    ref_seq: Array2<i64>,
    ref_bert: Array2<f32>,
    ref_audio_32k: Array2<f32>,
    ssl_content: ArrayBase<OwnedRepr<f32>, IxDyn>,
}

impl AsRef<Self> for ReferenceData {
    fn as_ref(&self) -> &Self {
        self
    }
}

pub struct GptSoVitsModel {
    text_processor: TextProcessor,
    sovits: Session,
    ssl: Session,
    t2s_encoder: Session,
    t2s_fs_decoder: Session,
    t2s_s_decoder: Session,
    num_layers: usize,
    run_options: RunOptions,
}

// --- KV Cache Configuration ---
/// Initial size for the sequence length of the KV cache.
const INITIAL_CACHE_SIZE: usize = 2048;
/// How much to increment the KV cache size by when reallocating.
const CACHE_REALLOC_INCREMENT: usize = 1024;

impl GptSoVitsModel {
    /// create new tts instance
    /// bert_path, g2pw_path and g2p_en_path can be None
    /// if bert path is none, the speech speed in chinese may become worse
    /// if g2pw path is none, the chinese speech quality may be worse
    /// g2p_en is still experimental, english speak quality may not be better because of bugs
    pub fn new<P>(
        sovits_path: P,
        ssl_path: P,
        t2s_encoder_path: P,
        t2s_fs_decoder_path: P,
        t2s_s_decoder_path: P,
        bert_path: Option<P>,
        g2pw_path: Option<P>,
        g2p_en_path: Option<P>,
    ) -> Result<Self, GSVError>
    where
        P: AsRef<Path>,
    {
        info!("Initializing TTSModel with ONNX sessions");

        let g2pw = G2PW::new(g2pw_path)?;

        let text_processor =
            TextProcessor::new(g2pw, G2pEn::new(g2p_en_path)?, BertModel::new(bert_path)?)?;

        Ok(GptSoVitsModel {
            text_processor,
            sovits: create_onnx_cpu_session(sovits_path)?,
            ssl: create_onnx_cpu_session(ssl_path)?,
            t2s_encoder: create_onnx_cpu_session(t2s_encoder_path)?,
            t2s_fs_decoder: create_onnx_cpu_session(t2s_fs_decoder_path)?,
            t2s_s_decoder: create_onnx_cpu_session(t2s_s_decoder_path)?,
            num_layers: NUM_LAYERS,
            run_options: RunOptions::new()?,
        })
    }

    pub async fn get_reference_data<P, S>(
        &mut self,
        reference_audio_path: P,
        ref_text: S,
        lang_id: LangId,
    ) -> Result<ReferenceData, GSVError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        info!("Processing reference audio and text: {}", ref_text.as_ref());
        let ref_text = ensure_punctuation(ref_text);
        let phones = self.text_processor.get_phone_and_bert(&ref_text, lang_id)?;
        let ref_seq: Vec<i64> = phones.iter().fold(Vec::new(), |mut seq, p| {
            seq.extend(p.1.clone());
            seq
        });

        let ref_bert: Vec<Array2<f32>> = phones.iter().map(|f| f.2.clone()).collect();
        // Concatenate along dimension 0
        let ref_bert = concatenate(
            Axis(0),
            &ref_bert.iter().map(|v| v.view()).collect::<Vec<_>>(),
        )?;

        let ref_seq = Array2::from_shape_vec((1, ref_seq.len()), ref_seq)?;
        let (ref_audio_16k, ref_audio_32k) = read_and_resample_audio(&reference_audio_path).await?;
        let ssl_content = self.process_ssl(&ref_audio_16k).await?;

        Ok(ReferenceData {
            ref_seq,
            ref_bert,
            ref_audio_32k,
            ssl_content,
        })
    }

    async fn process_ssl(
        &mut self,
        ref_audio_16k: &Array2<f32>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, GSVError> {
        let time = SystemTime::now();
        let ssl_output = self
            .ssl
            .run_async(
                inputs!["ref_audio_16k" => TensorRef::from_array_view(ref_audio_16k).unwrap()],
                &self.run_options,
            )?
            .await?;
        debug!("SSL processing time: {:?}", time.elapsed()?);
        Ok(ssl_output["ssl_content"]
            .try_extract_array::<f32>()?
            .into_owned())
    }

    /// Efficiently runs the streaming decoder loop with a pre-allocated, resizable KV cache.
    async fn run_t2s_s_decoder_loop(
        &mut self,
        sampler: &mut Sampler,
        sampling_param: SamplingParams,
        mut y_vec: Vec<i64>,
        mut k_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
        mut v_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
        prefix_len: usize,
        initial_valid_len: usize,
    ) -> Result<ArrayBase<OwnedRepr<i64>, IxDyn>, GSVError> {
        let mut idx = 0;
        let mut valid_len = initial_valid_len;
        y_vec.reserve(2048);

        loop {
            // --- 1. Prepare inputs using views of the valid cache portion ---
            let mut inputs = inputs![
                "iy" => TensorRef::from_array_view(unsafe {ArrayView2::from_shape_ptr((1, y_vec.len()), y_vec.as_ptr())})?,
                "y_len" => Tensor::from_array(Array::from_vec(vec![prefix_len as i64]))?,
                "idx" => Tensor::from_array(Array::from_vec(vec![idx as i64]))?,
            ];

            for i in 0..self.num_layers {
                // Create a view of the valid part of the cache
                let k = k_caches[i].slice(s![.., 0..valid_len, ..]).to_owned();
                let v = v_caches[i].slice(s![.., 0..valid_len, ..]).to_owned();

                inputs.push((
                    format!("ik_cache_{}", i).into(),
                    Tensor::from_array(k)?.into(),
                ));
                inputs.push((
                    format!("iv_cache_{}", i).into(),
                    Tensor::from_array(v)?.into(),
                ));
            }
            // --- 2. Run the decoder model for one step ---
            let mut output = self
                .t2s_s_decoder
                .run_async(inputs, &self.run_options)?
                .await?;

            let mut logits = output["logits"].try_extract_array_mut::<f32>()?;
            let mut logits = logits.as_slice_mut().unwrap().to_owned();

            if idx < 11 {
                logits.pop();
            }

            y_vec.push(sampler.sample(&mut logits, &y_vec, &sampling_param));

            let argmax_value = argmax(&logits);

            // --- 3. Check for reallocation and update caches ---
            let new_valid_len = valid_len + 1;

            // Check if we need to reallocate BEFORE writing to the new index.
            if new_valid_len > k_caches[0].shape()[1] {
                info!(
                    "Reallocating KV cache from {} to {}",
                    k_caches[0].shape()[1],
                    k_caches[0].shape()[1] + CACHE_REALLOC_INCREMENT
                );
                for i in 0..self.num_layers {
                    let old_k = &k_caches[i];
                    let old_v = &v_caches[i];

                    // Create new, larger arrays
                    let mut new_k_dims = old_k.raw_dim().clone();
                    new_k_dims[1] += CACHE_REALLOC_INCREMENT;
                    let mut new_v_dims = old_v.raw_dim().clone();
                    new_v_dims[1] += CACHE_REALLOC_INCREMENT;

                    let mut new_k = Array::zeros(new_k_dims);
                    let mut new_v = Array::zeros(new_v_dims);

                    // Copy existing valid data to the new arrays
                    new_k
                        .slice_mut(s![.., 0..valid_len, ..])
                        .assign(&old_k.slice(s![.., 0..valid_len, ..]));
                    new_v
                        .slice_mut(s![.., 0..valid_len, ..])
                        .assign(&old_v.slice(s![.., 0..valid_len, ..]));

                    // Replace the old caches with the new, larger ones
                    k_caches[i] = new_k;
                    v_caches[i] = new_v;
                }
            }

            // Update KV caches by pasting the newly generated slice of data
            for i in 0..self.num_layers {
                let inc_k_cache =
                    output[format!("k_cache_{}", i)].try_extract_array::<KvDType>()?;
                let inc_v_cache =
                    output[format!("v_cache_{}", i)].try_extract_array::<KvDType>()?;

                // The new data is the last row of the incremental output from the model
                let k_new_slice = inc_k_cache.slice(s![.., valid_len, ..]);
                let v_new_slice = inc_v_cache.slice(s![.., valid_len, ..]);

                // Paste the new row into our long-running cache at the correct position
                k_caches[i]
                    .slice_mut(s![.., valid_len, ..])
                    .assign(&k_new_slice);
                v_caches[i]
                    .slice_mut(s![.., valid_len, ..])
                    .assign(&v_new_slice);
            }

            // --- 4. Update valid length and check stop condition ---
            valid_len = new_valid_len;

            if idx >= 1500 || argmax_value == T2S_DECODER_EOS {
                let mut sliced = y_vec[(y_vec.len() - idx + 1)..(y_vec.len() - 1)]
                    .iter()
                    .map(|&i| if i == T2S_DECODER_EOS { 0 } else { i })
                    .collect::<Vec<i64>>();
                sliced.push(0);
                debug!(
                    "t2s final len: {}, prefix_len: {}",
                    sliced.len(),
                    prefix_len
                );
                let y = ArrayD::from_shape_vec(IxDyn(&[1, 1, sliced.len()]), sliced)?;
                return Ok(y);
            }
            idx += 1;
        }
    }

    /// synthesize async
    ///
    /// `text` is input text for run
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub async fn synthesize<R, S>(
        &mut self,
        text: S,
        reference_data: R,
        sampling_param: SamplingParams,
        lang_id: LangId,
    ) -> Result<impl Stream<Item = Result<Vec<f32>, GSVError>> + Send + Unpin, GSVError>
    where
        R: AsRef<ReferenceData>,
        S: AsRef<str>,
    {
        let time = SystemTime::now();
        let texts_and_seqs = self
            .text_processor
            .get_phone_and_bert(text.as_ref(), lang_id)?;
        debug!("g2pw and preprocess time: {:?}", time.elapsed()?);
        let ref_data = reference_data.as_ref().clone();

        let stream = stream! {
            for (text, seq, bert) in texts_and_seqs {
                debug!("process: {:?}", text);
                yield self.in_stream_once_gen(&text, &bert, &seq, &ref_data, sampling_param).await;
            }
        };

        Ok(Box::pin(stream))
    }

    async fn in_stream_once_gen(
        &mut self,
        _text: &str,
        text_bert: &Array2<f32>,
        text_seq_vec: &[i64],
        ref_data: &ReferenceData,
        sampling_param: SamplingParams,
    ) -> Result<Vec<f32>, GSVError> {
        let text_seq = Array2::from_shape_vec((1, text_seq_vec.len()), text_seq_vec.to_vec())?;
        let mut sampler = Sampler::new(VOCAB_SIZE);

        let prompts = {
            let time = SystemTime::now();
            let encoder_output = self
                .t2s_encoder
                .run_async(
                    inputs![
                        "ssl_content" => TensorRef::from_array_view(&ref_data.ssl_content)?
                    ],
                    &self.run_options,
                )?
                .await?;
            debug!("T2S Encoder time: {:?}", time.elapsed()?);
            encoder_output["prompts"]
                .try_extract_array::<i64>()?
                .into_owned()
        };

        let x = concatenate(Axis(1), &[ref_data.ref_seq.view(), text_seq.view()])?.to_owned();
        let bert = concatenate(
            Axis(1),
            &[
                ref_data.ref_bert.clone().permuted_axes([1, 0]).view(),
                text_bert.clone().permuted_axes([1, 0]).view(),
            ],
        )?;

        let bert = bert.insert_axis(Axis(0)).to_owned();

        let (mut y_vec, _) = prompts.clone().into_raw_vec_and_offset();

        let prefix_len = y_vec.len();

        let (y_vec, k_caches, v_caches, initial_seq_len) = {
            let time = SystemTime::now();
            let fs_decoder_output = self
                .t2s_fs_decoder
                .run_async(
                    inputs![
                        "x" => Tensor::from_array(x)?,
                        "prompts" => TensorRef::from_array_view(&prompts)?,
                        "bert" => Tensor::from_array(bert)?,
                    ],
                    &self.run_options,
                )?
                .await?;
            debug!("T2S FS Decoder time: {:?}", time.elapsed()?);

            let logits = fs_decoder_output["logits"]
                .try_extract_array::<f32>()?
                .into_owned();

            // --- Initialize large KV Caches ---
            // Get shape and initial data from the first-pass decoder.
            let k_init_first = fs_decoder_output["k_cache_0"].try_extract_array::<KvDType>()?;
            let initial_dims_dyn = k_init_first.raw_dim();
            let initial_seq_len = initial_dims_dyn[1];

            // Define the shape for our large, pre-allocated cache.
            let mut large_cache_dims = initial_dims_dyn.clone();
            large_cache_dims[1] = INITIAL_CACHE_SIZE;

            let mut k_caches = Vec::with_capacity(self.num_layers);
            let mut v_caches = Vec::with_capacity(self.num_layers);

            for i in 0..self.num_layers {
                let k_init =
                    fs_decoder_output[format!("k_cache_{}", i)].try_extract_array::<KvDType>()?;
                let v_init =
                    fs_decoder_output[format!("v_cache_{}", i)].try_extract_array::<KvDType>()?;

                // Create large, zero-initialized caches.
                let mut k_large = Array::zeros(large_cache_dims.clone());
                let mut v_large = Array::zeros(large_cache_dims.clone());

                // Copy the initial data from the first-pass decoder into the start of our large caches.
                k_large
                    .slice_mut(s![.., 0..initial_seq_len, ..])
                    .assign(&k_init);
                v_large
                    .slice_mut(s![.., 0..initial_seq_len, ..])
                    .assign(&v_init);

                k_caches.push(k_large);
                v_caches.push(v_large);
            }
            let (mut logits_vec, _) = logits.into_raw_vec_and_offset();
            logits_vec.pop(); // remove T2S_DECODER_EOS
            let sampling_rst = sampler.sample(&mut logits_vec, &y_vec, &sampling_param);
            y_vec.push(sampling_rst);
            (y_vec, k_caches, v_caches, initial_seq_len)
        };

        let time = SystemTime::now();
        let pred_semantic = self
            .run_t2s_s_decoder_loop(
                &mut sampler,
                sampling_param,
                y_vec,
                k_caches,
                v_caches,
                prefix_len,
                initial_seq_len,
            )
            .await?;
        debug!("T2S S Decoder all time: {:?}", time.elapsed()?);

        let time = SystemTime::now();
        let outputs = self
            .sovits
            .run_async(
                inputs![
                    "text_seq" => TensorRef::from_array_view(&text_seq)?,
                    "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                    "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?
                ],
                &self.run_options,
            )?
            .await?;
        debug!("SoVITS time: {:?}", time.elapsed()?);
        let output_audio = outputs["audio"].try_extract_array::<f32>()?;
        let (mut audio, _) = output_audio.into_owned().into_raw_vec_and_offset();
        for sample in &mut audio {
            *sample = *sample * 4.0;
        }
        // Find the maximum absolute value in the audio
        let max_audio = audio
            .iter()
            .filter(|&&x| x.is_finite()) // Ignore NaN or inf
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let audio = if max_audio > 1.0 {
            audio
                .into_iter()
                .map(|x| x / max_audio)
                .collect::<Vec<f32>>()
        } else {
            audio
        };

        Ok(audio)
    }
}

fn ensure_punctuation<S>(text: S) -> String
where
    S: AsRef<str>,
{
    if !text
        .as_ref()
        .ends_with(['。', '！', '？', '；', '.', '!', '?', ';'])
    {
        text.as_ref().to_owned() + "。"
    } else {
        text.as_ref().to_owned()
    }
}

fn resample_audio(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate {
        return input.to_owned();
    }

    UniformSourceIterator::new(SamplesBuffer::new(1, in_rate, input), 1, out_rate).collect()
}

async fn read_and_resample_audio<P>(path: P) -> Result<(Array2<f32>, Array2<f32>), GSVError>
where
    P: AsRef<Path>,
{
    let data = Cursor::new(read(path).await?);
    let decoder = Decoder::new(data)?;
    let sample_rate = decoder.sample_rate();
    let samples = if decoder.channels() == 1 {
        decoder.collect::<Vec<_>>()
    } else {
        UniformSourceIterator::new(decoder, 1, sample_rate).collect()
    };

    // Resample to 16kHz and 32kHz
    let mut ref_audio_16k = resample_audio(&samples, sample_rate, 16000);
    let ref_audio_32k = resample_audio(&samples, sample_rate, 32000);

    // Prepend 0.3 seconds of silence
    let silence_16k = vec![0.0; (0.3 * 16000.0) as usize]; // 8000 samples for 16kHz

    ref_audio_16k.splice(0..0, silence_16k);

    // Convert to Array2
    Ok((
        Array2::from_shape_vec((1, ref_audio_16k.len()), ref_audio_16k)?,
        Array2::from_shape_vec((1, ref_audio_32k.len()), ref_audio_32k)?,
    ))
}
