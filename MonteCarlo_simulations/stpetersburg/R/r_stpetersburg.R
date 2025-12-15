#' Simulate St Petersburg game payoffs
#'
#' Generates \code{n} independent draws from the St Petersburg distribution
#' where \eqn{P(X = 2^k) = 2^{-k}} for \eqn{k = 1, 2, \dots}.
#'
#' @param n Integer; number of samples to draw.
#' @param seed Optional integer seed for reproducibility. If \code{NULL}, the
#'   current RNG state is used.
#'
#' @return A numeric vector of length \code{n} containing the simulated payoffs.
#'
#' @details
#' The payoffs are constructed as \eqn{X = 2^K}, where \eqn{K} has a geometric
#' distribution with success probability 1/2. Note that \eqn{E[X] = \infty}.
#'
#' @examples
#' set.seed(1)
#' x <- r_stpetersburg(10)
#' summary(x)
#'
#' @export
r_stpetersburg <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  K = stats::rgeom(n, prob = 0.5) + 1
  2^K
}


#' Simulate multiple runs of the St Petersburg game
#'
#' Runs the St Petersburg game experiment \code{B} times (each with sample size \code{n}),
#' recording sums, scaled means, extreme-event counts, maxima, and robust estimators
#' for each run.
#'
#' @param n Integer; sample size (number of independent plays per run).
#' @param B Integer; number of independent simulation runs.
#' @param c Numeric; threshold factor for defining \eqn{N_n(c)}, the rare-event count
#'   (default 10).
#' @param trim_frac Numeric in [0, 0.5); fraction for symmetric trimming in the
#'   trimmed mean (default 0.1, i.e. 10\% trimming).
#' @param mom_blocks Integer; number of blocks for the median-of-means estimator
#'   (default 10).
#' @param seed Optional integer seed for reproducibility. If \code{NULL}, the
#'   current RNG state is used.
#'
#' @return A data frame with \code{B} rows, each corresponding to one simulation run,
#'   with columns:
#'   \describe{
#'     \item{\code{n}}{Sample size for that run (equals the input \code{n}).}
#'     \item{\code{S_n}}{Sum of the \code{n} payoff values in that run.}
#'     \item{\code{mean}}{Ordinary sample mean, \eqn{S_n / n}.}
#'     \item{\code{A_n}}{Scaled mean, \eqn{S_n / (n \log_2 n)}.}
#'     \item{\code{N_n}}{Count of observations \eqn{\ge c \cdot i \log_2 i} in that run.}
#'     \item{\code{M_n}}{Maximum payoff in that run.}
#'     \item{\code{trimmed}}{Trimmed mean of the \code{n} observations (using \code{trim_frac}).}
#'     \item{\code{median_of_means}}{Median-of-means estimate (using \code{mom_blocks} blocks).}
#'   }
#'
#' @examples
#' # Simulate 100 runs of sample size 1000
#' sim <- simulate_stpetersburg_experiment(n = 1000, B = 100, seed = 42)
#' head(sim)
#'
#' @export
simulate_stpetersburg_experiment <- function(n, B,
                                             c = 10,
                                             trim_frac = 0.1,
                                             mom_blocks = 10,
                                             seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  S_n_vec    = numeric(B)
  mean_vec   = numeric(B)
  A_n_vec    = numeric(B)
  N_n_vec    = numeric(B)
  M_n_vec    = numeric(B)
  trimmed_vec= numeric(B)
  mom_vec    = numeric(B)

  for (b in seq_len(B)) {
    X = r_stpetersburg(n)
    S_n = sum(X)

    S_n_vec[b]     = S_n
    mean_vec[b]    = S_n / n
    A_n_vec[b]     = S_n / (n * log2(n))

    i_seq          = seq_len(n)
    N_n_vec[b]     = sum(X >= c * i_seq * log2(i_seq))
    M_n_vec[b]     = max(X)
    trimmed_vec[b] = mean(X, trim = trim_frac)
    mom_vec[b]     = median_of_means(X, k = mom_blocks)
  }

  data.frame(
    n               = n,
    S_n             = S_n_vec,
    mean            = mean_vec,
    A_n             = A_n_vec,
    N_n             = N_n_vec,
    M_n             = M_n_vec,
    trimmed         = trimmed_vec,
    median_of_means = mom_vec
  )
}




#' Median-of-means estimator
#'
#' Computes the median-of-means estimator for a numeric vector \code{x}.
#' The data are split (in order) into \code{k} groups of approximately equal size;
#' the group means are computed and the median of those means is returned.
#'
#' @param x Numeric vector of data.
#' @param k Integer; number of blocks to use. If \code{k <= 1}, the ordinary
#'   sample mean is returned.
#'
#' @return Numeric scalar; the median-of-means estimate.
#'
#' @examples
#' x <- r_stpetersburg(1000, seed = 123)
#' median_of_means(x, k = 10)
#'
#' @export
median_of_means <- function(x, k) {
  n = length(x)
  if (k <= 1) {
    return(mean(x))
  }
  groups = rep_len(1:k, n)
  block_means = tapply(x, groups, mean)
  stats::median(block_means)
}






