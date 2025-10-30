# --- PREAMBLE ---
options(repos = c(CRAN = "https://cloud.r-project.org"))

install.packages("tidyverse", dependencies = TRUE)
install.packages("glmnet",   dependencies = TRUE)
install.packages("cowplot",  dependencies = TRUE)

library(tidyverse)
library(glmnet)
library(cowplot) # relevant libraries


# --- DATA ---
d <- read.csv("insurance.csv")

d$sex    <- factor(d$sex)
d$smoker <- factor(d$smoker)
d$region <- factor(d$region) # deals with categorical variables

d$region <- dplyr::recode(
  d$region,
  "southeast" = "SE",
  "southwest" = "SW",
  "northeast" = "NE",
  "northwest" = "NW" # otherwise titles go of page in plot
)


# --- TRAIN / TEST SPLIT ---
set.seed(123)
n   <- nrow(d)
idx <- sample(1:n, size = floor(0.8 * n)) #80/20 split
train <- d[idx, ]
test  <- d[-idx, ]


# --- MODEL MATRICES ---
X_train <- model.matrix(charges ~ ., data = train)[, -1, drop = FALSE] # design matrix
X_test  <- model.matrix(charges ~ ., data = test)[, -1, drop = FALSE]

eps     <- 1e-6 # so no log0 error
y_train <- log(train$charges + eps)  # modelled (log) target
y_test  <- test$charges              # not log for when calc errors


# --- GRID SEARCH OVER alpha AND lambda ---
alpha_grid <- c(0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)  # ridge -> lasso
nfolds_cv  <- 10

results <- tibble( #stores performance for each alpha/ lambda 
  alpha = numeric(),
  which_lambda = character(),   # "min" or "1se"
  lambda = numeric(),
  rmse = numeric(),
  mae = numeric(),
  cv_mse = numeric()
)

cv_objects <- vector("list", length(alpha_grid))  # create empty list for next part...

set.seed(123)

for (i in seq_along(alpha_grid)) {
  a <- alpha_grid[i]
  cv_fit <- cv.glmnet( # fit model for given alpha
    x = X_train, y = y_train,
    family = "gaussian",      # log-scale
    alpha = a,
    nfolds = nfolds_cv,
    type.measure = "mse",
    standardize = TRUE
  )
  cv_objects[[i]] <- cv_fit # save results to list


  for (choice in c("lambda.min", "lambda.1se")) { # test both lambda vals
    lam <- cv_fit[[choice]]

    log_pred <- as.numeric(predict(cv_fit, newx = X_test, s = lam, type = "response"))
    pred     <- pmax(exp(log_pred) - eps, 0) # to calc errors

    rmse <- sqrt(mean((pred - y_test)^2))
    mae  <- mean(abs(pred - y_test))
    cv_mse <- min(cv_fit$cvm)

    results <- add_row( # two rows for each alpha val in table
      results,
      alpha = a,
      which_lambda = if (choice == "lambda.min") "min" else "1se",
      lambda = lam,
      rmse = rmse,
      mae = mae,
      cv_mse = cv_mse
    )
  }
}

# --- PRINT BEST MODEL ---
results <- results |>
  arrange(rmse, mae) # lowest RMSE first, MAE tiebreak

best <- results %>% slice(1) #best model

cat("--- elastic-net tuning summary ---\n")
print(results, n = nrow(results))

cat("\nTHE BEST MODEL:\n")
print(best)

best_alpha   <- best$alpha # best params stored in variables
best_lambda  <- best$lambda
best_choice  <- best$which_lambda  # "min" or "1se"
best_rmse    <- best$rmse
best_mae     <- best$mae


# --- FIT FINAL MODEL WITH BEST alpha/ lambda ---
set.seed(123)

final_cv <- cv.glmnet(
  x = X_train, y = y_train,
  family = "gaussian",
  alpha = best_alpha,
  nfolds = nfolds_cv,
  type.measure = "mse",
  standardize = TRUE
)

log_pred_final <- as.numeric(predict(final_cv, newx = X_test, s = best_lambda, type = "response"))
pred_final     <- pmax(exp(log_pred_final) - eps, 0) # test set

rmse_final <- sqrt(mean((pred_final - y_test)^2))
mae_final  <- mean(abs(pred_final - y_test)) # errors

cat("\nelastic-net GLM [log-Gaussian]... FINAL MODEL\n")
cat("alpha =", best_alpha,
    "| lambda*", "(", best_choice, ") =", signif(best_lambda, 5),
    "| RMSE =", round(rmse_final, 2),
    "| MAE =", round(mae_final, 2), "\n")


# --- PLOT... predicted vs actual ---
results_plot <- data.frame(actual = y_test, predicted = pred_final)

p_scatter <- ggplot(results_plot, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5, colour = "#8f07e3") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "#000000") +
  labs(title = NULL, x = "Actual Charges", y = "Predicted Charges") +
  annotate(
    "text",
    x = Inf, y = -Inf,
    label = paste0(
      "RMSE = ", round(rmse_final, 1),
      "\nMAE  = ", round(mae_final, 1),
      "\nalpha = ", best_alpha,
      "\nλ (", best_choice, ") = ", signif(best_lambda, 5)
    ),
    hjust = 1.1, vjust = -0.5,
    size = 4, fontface = "bold", colour = "#000000"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
    plot.title = element_blank(),
    plot.margin = margin(10, 20, 10, 10)
  )

ggsave("predictedVSactualPremiumsElNetGLM.png",
       plot = p_scatter,
       width = 7, height = 5, units = "in")


# --- VIEW COEFFS ---
coefs_sp <- coef(final_cv, s = best_lambda)  # sparse S4 (dgCMatrix)
coefs    <- as.matrix(coefs_sp)

coef_tbl <- tibble::tibble(
  term     = rownames(coefs),
  estimate = as.numeric(coefs[, 1])
) |>
  dplyr::arrange(dplyr::desc(abs(estimate)))

cat("\nNon-zero coefficients (alpha =", best_alpha, ", lambda =", signif(best_lambda, 5), "):\n")
print(coef_tbl, n = nrow(coef_tbl))