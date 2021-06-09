data {
  int<lower=0> T;
  vector[T] x;  
}
parameters {
  real<lower=-1, upper=1> phi;  
  real mu;
  real<lower=0> sigma;
}
model {
  for (t in 2:T)
    x[t] ~ normal(mu + phi * ( x[t-1] - mu ), sigma);
}
