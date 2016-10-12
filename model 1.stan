data {
  
  int<lower=0> N_obs;
  vector<lower=0> [N_obs] loss;
  
  matrix [N_obs, 14] x_cont;

  //groups
  int D;
  //vector<upper=D> [N_obs] x_cat;
  int<upper=D> x_cat[N_obs];

}
parameters {
  
  //regression coefficient
  vector[14] beta[D];
  
  real<lower=0> sigma;

}

model {
  
  {
    
    vector [N_obs] mu;
    
    for(obs_i in 1:N_obs) {
      int group_i;
      mu[obs_i] = x_cont[obs_i] * beta[x_cat[obs_i]];
      
    }
  
    sigma ~ cauchy(0, 1);
    loss ~ lognormal(mu, sigma);
    
  }
  
}