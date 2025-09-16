use {
    rand::{
        SeedableRng,
        distr::{Distribution, weighted::WeightedIndex},
        rngs::StdRng,
    },
    std::{cmp::Ordering, collections::HashSet},
};

/// Finds the token with the highest logit value (argmax).
pub fn argmax(logits: &[f32]) -> i64 {
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_idx = 0;

    for (idx, &logit) in logits.iter().enumerate() {
        if logit > max_logit {
            max_logit = logit;
            max_idx = idx;
        }
    }
    max_idx as i64
}

// Sampling parameters (unchanged)
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
}

// Builder for SamplingParams (unchanged)
pub struct SamplingParamsBuilder {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: f32,
}

impl SamplingParams {
    pub fn builder() -> SamplingParamsBuilder {
        SamplingParamsBuilder::new()
    }
}

impl SamplingParamsBuilder {
    fn new() -> Self {
        SamplingParamsBuilder {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = if temperature >= 0.0 { temperature } else { 1.0 };
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = if repetition_penalty > 0.0 {
            repetition_penalty
        } else {
            1.0
        };
        self
    }

    pub fn build(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
        }
    }
}

/// Processes logits to sample a token ID, applying various strategies
/// like temperature, repetition penalty, and Top-K/Top-P sampling.
pub struct Sampler {
    rng: StdRng,
    /// Reusable buffer for probabilities to avoid re-allocation in the sampling loop.
    probs: Vec<f32>,
}

unsafe impl Send for Sampler {}

impl Sampler {
    /// Creates a new Sampler.
    ///
    /// # Arguments
    /// * `vocab_size`: The size of the vocabulary, used to pre-allocate buffers for efficiency.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            rng: StdRng::from_os_rng(),
            probs: Vec::with_capacity(vocab_size),
        }
    }

    /// Applies a penalty to the logits of repeated tokens.
    fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[i64], penalty: f32) {
        if penalty == 1.0 {
            return;
        }
        let prev_tokens_set: HashSet<_> = prev_tokens.iter().copied().collect();
        for (token_id, logit) in logits.iter_mut().enumerate() {
            if prev_tokens_set.contains(&(token_id as i64)) {
                if *logit >= 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }
    }

    /// Applies temperature scaling to the logits.
    fn apply_temperature(logits: &mut [f32], temperature: f32) {
        if temperature > 0.0 {
            let inv_temp = 1.0 / temperature;
            for logit in logits.iter_mut() {
                *logit *= inv_temp;
            }
        }
    }

    /// Computes the softmax of logits and stores the result in the internal `probs` buffer.
    fn softmax(&mut self, logits: &[f32]) {
        self.probs.clear();
        if logits.is_empty() {
            return;
        }

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0;
        self.probs.extend(logits.iter().map(|&logit| {
            let exp_val = (logit - max_logit).exp();
            sum_exp += exp_val;
            exp_val
        }));

        if sum_exp > 0.0 {
            let inv_sum_exp = 1.0 / sum_exp;
            for prob in self.probs.iter_mut() {
                *prob *= inv_sum_exp;
            }
        }
    }

    /// Main sampling method with performance optimizations.
    pub fn sample(
        &mut self,
        logits: &mut [f32],
        prev_tokens: &[i64],
        params: &SamplingParams,
    ) -> i64 {
        Self::apply_repetition_penalty(logits, prev_tokens, params.repetition_penalty);

        // Optimized path for greedy decoding (argmax).
        if params.temperature == 0.0 {
            return argmax(logits);
        }

        Self::apply_temperature(logits, params.temperature);
        self.softmax(logits);

        let mut candidates: Vec<(usize, f32)> = self.probs.iter().copied().enumerate().collect();

        if candidates.is_empty() {
            return argmax(logits);
        }

        // --- Top-K Filtering (Optimized O(V) selection) ---
        if let Some(k) = params.top_k {
            if k > 0 && k < candidates.len() {
                candidates.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                candidates.truncate(k);
            }
        }

        // --- Top-P (Nucleus) Filtering (on at most K candidates) ---
        if let Some(p) = params.top_p {
            if p < 1.0 {
                candidates
                    .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let mut cum_prob = 0.0;
                let mut cutoff = candidates.len();
                for (i, &(_, prob)) in candidates.iter().enumerate() {
                    cum_prob += prob;
                    if cum_prob >= p {
                        cutoff = i + 1;
                        break;
                    }
                }
                candidates.truncate(cutoff);
            }
        }

        // --- Final Sampling ---
        let weights = candidates.iter().map(|&(_, p)| p);
        let dist = match WeightedIndex::new(weights) {
            Ok(d) => d,
            Err(_) => {
                // Fallback if distribution fails (e.g., all probs are 0 after filtering).
                // Return the highest probability candidate before this step.
                return candidates
                    .first()
                    .map_or_else(|| argmax(logits), |&(idx, _)| idx as i64);
            }
        };

        let sampled_candidate_index = dist.sample(&mut self.rng);
        candidates[sampled_candidate_index].0 as i64
    }
}
