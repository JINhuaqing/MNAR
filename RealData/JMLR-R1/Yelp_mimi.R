library(mimi)
library(reticulate)
np <- import("numpy")
library(ROCR)
library(parallel)


mimi.logi.fn <- function(Y, X, R, CT, Cb, alp.init, theta.init, trace.it=T, maxit=100){
  p <- ncol(X)
  m <- dim(Y)[1]
  n <- dim(Y)[2]
  var.type <- rep("binomial", n)
  Y.miss <- Y
  Y.miss[R==0] <- NA
  res <- mimi(Y.miss, model="c", x=X, var.type=var.type, lambda1=CT, lambda2 = Cb, 
              alpha0=alp.init, theta0=theta.init, algo="bc", maxit=maxit, trace.it = trace.it, max.rank=10)
  return(res)
}

post.mimi.fn <- function(res){
    probs <-  exp(res$param)/(1+exp(res$param))
    return(probs)
}

probs2auc <- function(probs, Ynp, rYnp, Rnp){
    idxMat <- rYnp != -1 & Rnp == 0
    obs <- Ynp[idxMat]
    eProbs <- probs[idxMat]
    pred <- prediction(eProbs, obs)
    auc.res <- performance(pred, "auc")
    return(auc.res@y.values)
}

aucs2Idxs <- function(aucs){
    Cb.idx <- ifelse(which.max(aucs) %% 3==0, which.max(aucs) %/% 3, which.max(aucs) %/% 3 + 1)
    CT.idx <- ifelse(which.max(aucs) %% 3==0, 3, which.max(aucs) %% 3)
    return(c(Cb.idx, CT.idx))
}



Ynp <- np$load("./npData/Y.npz")
Xnp <- np$load("./npData/X.npz")
rYnp <- np$load("./npData/rY.npz")
m = dim(Xnp)[1]
n = dim(Xnp)[2]
p = dim(Xnp)[3]
matX <- matrix(Xnp, nrow=m*n)




# OR
OR <- 60
Rnp.ps <- paste0("./npData/R_", OR, "_", 1:20, ".npz")
Rnp0 <- np$load(Rnp.ps[1])
CTs <- c(1, 1e2, 1e3)
Cbs <- c(1, 1e2, 1e3)

# test time
# alp.init <- rep(0, p)
# theta.init <- matrix(rnorm(n*m), nrow=m) + 0.1
# t0 <- Sys.time()
# res <- mimi.logi.fn(Ynp, matX, Rnp0, CT=1, Cb=1, alp.init=alp.init, theta.init=theta.init)
# t1 <- Sys.time()
# print(t1-t0)

# Tuning Cb, CT
alp.init <- rep(0, p)
theta.init <- matrix(rnorm(n*m), nrow=m) + 0.1
aucs <- c()
for (Cb in Cbs){
    for (CT in CTs){
        res <- mimi.logi.fn(Ynp, matX, Rnp0, CT=CT, Cb=Cb, alp.init=alp.init, theta.init=theta.init)
        probs <- post.mimi.fn(res)
        cur.auc <- probs2auc(probs, Ynp, rYnp, Rnp0) 
        aucs <- c(aucs, cur.auc)
    }
}

idxs <- aucs2Idxs(aucs)
selCb <- Cbs[idxs[1]]
selCT <- CTs[idxs[2]]

runfn <- function(idx){
    Rnp.p <- Rnp.ps[idx]
    print(Rnp.p)
    Rnp <- np$load(Rnp.p)
    alp.init <- rep(0, p)
    theta.init <- matrix(rnorm(n*m), nrow=m) + 0.1
    res <- mimi.logi.fn(Ynp, matX, Rnp0, CT=selCT, Cb=selCb, alp.init=alp.init, theta.init=theta.init)
    probs <- post.mimi.fn(res)
    cur.auc <- probs2auc(probs, Ynp, rYnp, Rnp) 
    res <- list()
    res$v <- cur.auc
    res$idx <- idx
    res
}

# Simulation
# res.aucs <- c()
# for (Rnp.p in Rnp.ps){
#     print(Rnp.p)
#     Rnp <- np$load(Rnp.p)
#     alp.init <- rep(0, p)
#     theta.init <- matrix(rnorm(n*m), nrow=m) + 0.1
#     res <- mimi.logi.fn(Ynp, matX, Rnp0, CT=selCT, Cb=selCb, alp.init=alp.init, theta.init=theta.init)
#     probs <- post.mimi.fn(res)
#     cur.auc <- probs2auc(probs, Ynp, rYnp, Rnp) 
#     res.aucs <- c(res.aucs, cur.auc)
# }

res.aucs <- mclapply(1:20, runfn, mc.cores=4)
fName <- paste0("./", "MIMI_", OR, ".RData")
save(res.aucs, file=fName)
