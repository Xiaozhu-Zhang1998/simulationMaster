#' @title Shiny App of Simulation Master
#'
#' @description This is a function to run the Shiny App of the "Simulation Master".
#' This Shiny App contains two menus, one of which is regarding regression:
#' it provides different parameters to generate sparse data, and also parameters to tune the Lasso/Ridge/Elastic Net;
#' the other of which is regarding classification: it provides the generation functionality of
#' four types of data, and tuning parameters of kernel SVM as well. This Shiny App focus on the
#' feature selection of regression problems and the prediction performance of classification problems.
#'
#' @note This is simply a function to trigger the Shiny App, so it bears no parameters or values.
#'
#' @examples
#' \dontrun{
#' run_simulation_master()
#' }
#'
#' @import shinydashboard
#' @import shiny
#' @importFrom bestsubset sim.xy
#' @importFrom glmnet cv.glmnet
#' @importFrom mvtnorm rmvnorm
#' @import patchwork
#' @importFrom shinycssloaders withSpinner
#' @importFrom kernlab ksvm predict
#' @import ggplot2
#' @importFrom tippy tippy_this
#' @import tidyr
#' @import dplyr
#' @importFrom stats coef qchisq rnorm
#'
#' @export

run_simulation_master = function(){
  shinyApp(
    # ui
    ui = dashboardPage(
      dashboardHeader(
        title = "Simulation Master", titleWidth = 230, disable = FALSE,
        tags$li(class = "dropdown", tags$a(href = "https://github.com/Xiaozhu-Zhang1998/simulationMaster",
                                           icon("github"), "About", target = "_blank"))
      ),

      dashboardSidebar(
        sidebarMenu(
          menuItem(text = "Regression: Glmnet", tabName = "reg", icon = icon("bullseye")),
          menuItem(text = "Classification: Kernel SVM", tabName = "cla", icon = icon("flag"))
        )
      ),

      dashboardBody(
        tabItems(
          tabItem(tabName = "reg", uiOutput("regpage")),
          tabItem(tabName = "cla", uiOutput("claspage"))
        )
      )

    ),

    # server
    server = function(input, output, session){
      # regression page ----
      output$regpage = renderUI({
        f1 = fluidRow(
          box(title = "Data Generation Parameters", width = 6, solidHeader = T, status = "success",
              collapsible = TRUE,
              fluidRow(
                column(6,
                       sliderInput("n", "Training set size", min = 100, max = 10000, step = 10, value = 2000),
                       sliderInput("p", "Number of features", min = 100, max = 1000, step = 1, value = 500),
                       p(HTML("<b>Correlation coefficient (rho)</b>"), span(shiny::icon("info-circle"), id = "info_rho"),
                         sliderInput("rho", NULL, min = 0, max = 1, value = 0.5),
                         tippy_this(elementId = "info_rho",
                                    tooltip = "The correlation coefficient of Toeplitz covariance matrix",
                                    placement = "right")
                       ),
                       p(HTML("<b>Beta Type</b>"), span(shiny::icon("info-circle"), id = "info_beta_type"),
                         selectInput("beta.type", NULL, choices = c(1:5)),
                         tippy_this(elementId = "info_beta_type",
                                    tooltip = "See the paper by Hastie, Tibshirani & Tibshirani, 2020",
                                    placement = "right")
                       )

                ),

                column(6,
                       sliderInput("nval", "Test set size", min = 1, max = 10000, step = 10, value = 2000),
                       sliderInput("s", "Number of non-zero coefficients", min = 1, max = 50, step = 10, value = 30),
                       sliderInput("snr", "SNR", min = 0.05, max = 6, value = 3),
                       sliderInput("fold", "Number of folds for CV", min = 5, max = 10, value = 5)
                ))

          ),

          box(title = "Model Specification Parameters", width = 6, solidHeader = T, status = "success",
              collapsible = TRUE,
              fluidRow(
                column(6,
                       p(HTML("<b>Alpha</b>"), span(shiny::icon("info-circle"), id = "info_alpha"),
                         sliderInput("alpha", NULL, min = 0, max = 1, value = 0.5),
                         tippy_this(elementId = "info_alpha",
                                    tooltip = "The 'alpha' parameter in the function 'Glmnet'. Being 0 leads to Ridge. Being 1 leads to LASSO.",
                                    placement = "right")
                       ),
                       sliderInput("min.log.lambda", "Min of log(lambda)", min = -10, max = -1, value = -3),

                ),
                column(6,
                       sliderInput("nlambda", "Number of lambdas", min = 10, max = 100, value = 50),
                       sliderInput("max.log.lambda", "Max of log(lambda)", min = 1, max = 10, value = 3),
                ))
          )
        )

        f2 = fluidRow(
          box(title = "Training & CV MSE (Please select a region on the plot to view feature selection results)",
              width = 9, solidHeader = TRUE, collapsible = TRUE, status = "warning",
              plotOutput("regtraintest", brush = brushOpts(id = "plot_brush")) %>% withSpinner(type = 5)
          ),

          box(title = "Summary", width = 3, solidHeader = TRUE, collapsible  = TRUE,
              status = "warning",
              uiOutput("regsum") %>% withSpinner(type = 1))
        )

        f3 = fluidRow(
          tabBox(title = "Feature Selection", width = 12,
                 tabPanel("Statistics", tableOutput("featurestat") %>% withSpinner(type = 1)),
                 tabPanel("Selection Plots", plotOutput("feature") %>% withSpinner(type = 5))
          )
        )

        return(list(f1, f2, f3))
      })


      regdat = reactive({
        sim.xy(n = input$n, p = input$p, nval = input$nval,
               rho = input$rho, s = input$s, beta.type = as.numeric(input$beta.type),
               snr = input$snr)
      })

      regfit = reactive({
        lambda = exp(seq(from = input$min.log.lambda, to = input$max.log.lambda,
                         length = input$nlambda))
        cv.glmnet(x = regdat()$x, y = regdat()$y, family = "gaussian",
                  alpha = input$alpha, lambda = lambda, type.measure = "mse", nfolds = input$fold)
      })

      regmse = reactive({
        fit = regfit()$glmnet.fit
        train = apply(
          (regdat()$x %*% as.matrix(fit$beta) - regdat()$y %*% t(rep(1, input$nlambda))) ** 2,
          2,
          mean
        )
        val = regfit()$cvm
        sd = regfit()$cvsd
        tibble(Train = train, Validation = val, sd = sd, loglambda = log(regfit()$lambda)) %>%
          pivot_longer(cols = c("Train", "Validation"), names_to = "Type", values_to = "MSE") %>%
          mutate_(sd = ~ ifelse(Type == "Validation", sd, NA))
      })

      output$regtraintest = renderPlot({
        ggplot(data = regmse(), aes_(x = ~ loglambda, y = ~ MSE, color = ~ Type)) +
          geom_point() +
          geom_line() +
          geom_errorbar(aes_(ymin = ~ MSE - sd, ymax = ~ MSE + sd)) +
          labs(x = "log(lambda)", y = "MSE",
               caption = "Black line indicates the lambda achieving minimal validation MSE,
                            and red line indicates the lambda achieving one-standard-error MSE.") +
          geom_vline(xintercept = log(regfit()$lambda.min), color = "black") +
          geom_vline(xintercept = log(regfit()$lambda.1se), color = "red") +
          theme_bw()
      })

      output$regsum = renderUI({
        pred = predict(regfit()$glmnet.fit, newx = regdat()$xval, type = c("response"))
        test = apply(
          (pred - regdat()$yval %*% t(rep(1, input$nlambda))) ** 2,
          2,
          mean
        )

        l1 = paste0("The lambda that achieves the minimal validation MSE is ",
                    round(regfit()$lambda.min, 4), ", and its logarithm is ",
                    round(log(regfit()$lambda.min), 4), ". The corresponding test MSE is ",
                    round(test[log_black_id()], 4), ". ")
        l2 = hr()
        l3 = paste0("The lambda that identifies the one-standard-error MSE is ",
                    round(regfit()$lambda.1se, 4), ", and its logarithm is ",
                    round(log(regfit()$lambda.1se), 4), ". The corresponding test MSE is ",
                    round(test[log_red_id()], 4), ". ")
        return(list(l1, l2, l3))
      })

      s.dat = reactive({
        req(input$plot_brush)
        brushedPoints(regmse(), input$plot_brush)
      })

      trueid = reactive({
        if(input$beta.type == "5") {
          trueid = rep("Null beta", input$p)
          trueid[1:input$s] = "True beta"
          return(trueid)
        }
        else {
          trueid = ifelse(regdat()$beta == 0, "Null beta", "True beta")
          return(trueid)
        }
      })

      log_lambda = reactive({ log(regfit()$lambda) })
      beta = reactive({ as.matrix(regfit()$glmnet.fit$beta) })
      log_min_lambda = reactive({ s.dat() %>% pull(c("loglambda")) %>% min() })
      log_min_id = reactive({ which(log_lambda() == log_min_lambda()) })
      log_max_lambda = reactive({ s.dat() %>% pull(c("loglambda")) %>% max() })
      log_max_id = reactive({ which(log_lambda() == log_max_lambda()) })
      log_black_lambda = reactive({ log(regfit()$lambda.min) })
      log_black_id = reactive({ which(log_lambda() == log_black_lambda()) })
      log_red_lambda = reactive({ log(regfit()$lambda.1se) })
      log_red_id = reactive({ which(log_lambda() == log_red_lambda()) })
      log_black_beta = reactive({ beta()[, log_black_id()] })
      log_red_beta = reactive({ beta()[, log_red_id()] })
      log_min_beta = reactive({ beta()[, log_min_id()] })
      log_max_beta = reactive({ beta()[, log_max_id()] })

      output$feature = renderPlot({
        req(input$plot_brush)
        if(nrow(s.dat()) == 0) return(NULL)

        # fit1
        p1 = ggplot(tibble(feature = names(log_black_beta()), coef = log_black_beta(),
                           trueid = trueid()),
                    aes_(x = ~ feature, y = ~ coef, fill = ~ trueid)) +
          geom_bar(stat = "identity") +
          labs(y = "Coefficients", x = "Features", fill = "",
               title = paste0("Black line: log(lambda) = ", round(log_black_lambda(), 4))) +
          theme_bw() +
          theme(axis.text.x = element_blank())
        # fit2
        p2 = ggplot(tibble(feature = names(log_red_beta()), coef = log_red_beta(),
                           trueid = trueid()),
                    aes_(x = ~ feature, y = ~ coef, fill = ~ trueid)) +
          geom_bar(stat = "identity") +
          labs(y = "Coefficients", x = "Features", fill = "",
               title = paste0("Red line: log(lambda) = ", round(log_red_lambda(), 4))) +
          theme_bw() +
          theme(axis.text.x = element_blank())
        # fit3

        p3 = ggplot(tibble(feature = names(log_min_beta()), coef = log_min_beta(),
                           trueid = trueid()),
                    aes_(x = ~ feature, y = ~ coef, fill = ~ trueid)) +
          geom_bar(stat = "identity") +
          labs(y = "Coefficients", x = "Features", fill = "",
               title = paste0("Min selected lambda: log(lambda) = ",
                              round(log_min_lambda(), 4))) +
          theme_bw() +
          theme(axis.text.x = element_blank())
        # fit4

        p4 = ggplot(tibble(feature = names(log_max_beta()), coef = log_max_beta(),
                           trueid = trueid()),
                    aes_(x = ~ feature, y = ~ coef, fill = ~ trueid)) +
          geom_bar(stat = "identity") +
          labs(y = "Coefficients", x = "Features", fill = "",
               title = paste0("Max selected lambda: log(lambda) = ",
                              round(log_max_lambda(), 4))) +
          theme_bw() +
          theme(axis.text.x = element_blank())
        return(p1 + p2 + p3 + p4 + plot_layout(ncol = 2))
      })


      output$featurestat = renderTable({
        req(input$plot_brush)
        if(nrow(s.dat()) == 0) return(NULL)

        tibble(
          Model = c("Black line", "Red line", "Min selected lambda", "Max selected lambda"),
          `log(lambda)` = c(log_black_lambda(), log_red_lambda(), log_min_lambda(), log_max_lambda()),
          TP = c(sum(log_black_beta() != 0 & trueid() != "Null beta"),
                 sum(log_red_beta() != 0 & trueid() != "Null beta"),
                 sum(log_min_beta() != 0 & trueid() != "Null beta"),
                 sum(log_max_beta() != 0 & trueid() != "Null beta")),
          FP = c(sum(log_black_beta() != 0 & trueid() == "Null beta"),
                 sum(log_red_beta() != 0 & trueid() == "Null beta"),
                 sum(log_min_beta() != 0 & trueid() == "Null beta"),
                 sum(log_max_beta() != 0 & trueid() == "Null beta")),
          TN = c(sum(log_black_beta() == 0 & trueid() == "Null beta"),
                 sum(log_red_beta() == 0 & trueid() == "Null beta"),
                 sum(log_min_beta() == 0 & trueid() == "Null beta"),
                 sum(log_max_beta() == 0 & trueid() == "Null beta")),
          FN = c(sum(log_black_beta() == 0 & trueid() != "Null beta"),
                 sum(log_red_beta() == 0 & trueid() != "Null beta"),
                 sum(log_min_beta() == 0 & trueid() != "Null beta"),
                 sum(log_max_beta() == 0 & trueid() != "Null beta"))
        ) %>%
          mutate_(Accuracy = ~ (TP + TN) / (TP + FP + TN + FN),
                  TPR = ~ TP / (TP + FN),
                  FPR = ~ FP / (FP + TN),
                  Precision = ~ TP / (TP + FP),
                  FDR = ~ FP / (FP + TP),
                  `F-score` = ~ 2 * TPR * Precision / (TPR + Precision)
          )
      })




      # classification page ----
      output$claspage = renderUI({
        f1 = fluidRow(
          box(title = "Data Generation Parameters", width = 6, solidHeader = T, status = "success",
              collapsible = TRUE,
              fluidRow(
                column(6,
                       sliderInput("cls.n", "Training set size", min = 100, max = 1000, step = 10, value = 500),
                       selectInput("data.type", "Data Type",
                                   choices = c("Circle", "XOR", "Gaussian", "Spiral"))
                ),

                column(6,
                       sliderInput("cls.nval", "Test set size", min = 100, max = 1000, step = 10, value = 500),
                )),

              fluidRow(
                column(12, uiOutput("dynamic.input"))
              )

          ),

          box(title = "Kernel SVM Parameters", width = 6, solidHeader = T, status = "success",
              collapsible = TRUE,
              fluidRow(
                column(6,
                       sliderInput("c", "Cost", min = 0, max = 100, value = 10),
                       p(HTML("<b>Kernel</b>"), span(shiny::icon("info-circle"), id = "info_circle_kernel"),
                         selectInput("kernel", NULL,
                                     choices = c("RBF", "Polynomial", "Linear", "Hyperbolic tangent",
                                                 "Laplacian", "Bessel", "ANOVA RBF", "Spline")),
                         tippy_this(elementId = "info_circle_kernel",
                                    tooltip = "Based on the function 'ksvm' in the package 'kernlab'",
                                    placement = "right")
                       )

                ),
                column(6,
                       sliderInput("cross", "Number of folds for CV", min = 5, max = 10, value = 5)
                )),
              fluidRow(
                column(12,
                       uiOutput("dynamic.kernel"))
              )
          )
        )

        f2 = fluidRow(
          box(title = "Decision boundries", width = 8, solidHeader = TRUE, collapsible = TRUE,
              status = "warning",
              plotOutput("clasplot") %>% withSpinner(type = 5)
          ),

          box(title = "Summary", width = 4, solidHeader = TRUE, collapsible  = TRUE,
              status = "warning",
              uiOutput("classum") %>% withSpinner(type = 1))
        )

        return(list(f1, f2))
      })


      output$dynamic.input = renderUI({
        # circle
        if(input$data.type == "Circle") {
          f = fluidRow(
            column(
              6,
              p(HTML("<b>Prob. of Chi-square</b>"), span(shiny::icon("info-circle"), id = "info_circle_p"),
                sliderInput("circle.p", NULL, min = 0.1, max = 0.9, value = 0.5, step = 0.05),
                tippy_this(elementId = "info_circle_p",
                           tooltip = "Chi-square is the threshold for square distance to origin",
                           placement = "right")
              ),
              p(HTML("<b>Df. of Chi-square</b>"), span(shiny::icon("info-circle"), id = "info_circle_df"),
                sliderInput("circle.df", NULL, min = 1, max = 8, value = 1, step = 0.5),
                tippy_this(elementId = "info_circle_df",
                           tooltip = "Chi-square is the threshold for square distance to origin",
                           placement = "right")
              )
            ),
            column(
              6,
              sliderInput("circle.noise", "Noise", min = 0, max = 1, value = 0.1)
            )
          )
          return(f)
        }

        # XOR
        if(input$data.type == "XOR") {
          f = fluidRow(
            column(
              6,
              sliderInput("xor.sigma", "Variability of data", min = 1, max = 3, value = 1, step = 0.1),
            ),
            column(
              6,
              sliderInput("xor.noise", "Noise", min = 0, max = 1, value = 0.1)
            )
          )
          return(f)
        }

        # Gaussian
        if(input$data.type == "Gaussian") {
          f = fluidRow(
            column(
              6,
              sliderInput("gaussian.1.mu1", "Mu1 of class 1", min = -3, max = 1, value = -1, step = 0.1),
              sliderInput("gaussian.2.mu1", "Mu1 of class 2", min = -1, max = 3, value = 1, step = 0.1),
              sliderInput("gaussian.1.sigma1", "Sigma1 of class 1", min = 0.1, max = 2, value = 0.5, step = 0.1),
              sliderInput("gaussian.2.sigma1", "Sigma1 of class 2", min = 0.1, max = 2, value = 0.5, step = 0.1),
              sliderInput("gaussian.1.rho", "Rho of class 1", min = -1, max = 1, value = 0.3, step = 0.1)
            ),
            column(
              6,
              sliderInput("gaussian.1.mu2", "Mu2 of class 1", min = -3, max = 1, value = -1, step = 0.1),
              sliderInput("gaussian.2.mu2", "Mu2 of class 2", min = -1, max = 3, value = 1, step = 0.1),
              sliderInput("gaussian.1.sigma2", "Sigma2 of class 1", min = 0.1, max = 2, value = 0.5, step = 0.1),
              sliderInput("gaussian.2.sigma2", "Sigma2 of class 2", min = 0.1, max = 2, value = 0.5, step = 0.1),
              sliderInput("gaussian.2.rho", "Rho of class 2", min = -1, max = 1, value = 0.3, step = 0.1)
            )
          )
          return(f)
        }

        if(input$data.type == "Spiral") {
          f = fluidRow(
            column(
              6,
              sliderInput("spiral.r", "Radius", min = 0, max = 3, value = 1, step = 0.01),
              sliderInput("spiral.theta1", "Angle of class 1", min = 0, max = 40, value = c(4, 12))
            ),
            column(
              6,
              sliderInput("spiral.noise", "Noise", min = 0, max = 1, value = 0.1),
              sliderInput("spiral.theta2", "Angle of class 2", min = 0, max = 40, value = c(15, 20))
            )
          )
          return(f)
        }

      })


      output$dynamic.kernel = renderUI({
        # RBF or Laplacian
        if(input$kernel == "RBF" | input$kernel == "Laplacian") {
          f = fluidRow(
            column(
              6,
              sliderInput("rbf.sigma", "Sigma", min = 0, max = 10, value = 0.5, step = 0.1)
            )
          )
          return(f)
        }
        # Polynomial
        if(input$kernel == "Polynomial") {
          f = fluidRow(
            column(
              6,
              sliderInput("poly.degree", "Degree", min = 1, max = 10, value = 2, step = 1),
              sliderInput("poly.offset", "Offset", min = 0, max = 20, value = 1, step = 0.1)
            ),
            column(
              6,
              sliderInput("poly.scale", "Scale", min = 1, max = 10, value = 2, step = 1)
            )
          )
          return(f)
        }
        # Hyperbolic
        if(input$kernel == "Hyperbolic tangent") {
          f = fluidRow(
            column(
              6,
              sliderInput("hyperbolic.scale", "Scale", min = 1, max = 10, value = 2, step = 1)
            ),
            column(
              6,
              sliderInput("hyperbolic.offset", "Offset", min = 0, max = 20, value = 1, step = 0.1)
            )
          )
          return(f)
        }
        # Bessel
        if(input$kernel == "Bessel") {
          f = fluidRow(
            column(
              6,
              sliderInput("bessel.sigma", "Scale", min = 0, max = 10, value = 0.5, step = 0.1),
              sliderInput("bessel.degree", "Degree", min = 1, max = 10, value = 2, step = 1)
            ),
            column(
              6,
              sliderInput("bessel.order", "Order", min = 1, max = 10, value = 2, step = 1)
            )
          )
          return(f)
        }
        # ANOVA RBF
        if(input$kernel == "ANOVA RBF") {
          f = fluidRow(
            column(
              6,
              sliderInput("anova.sigma", "Scale", min = 0, max = 10, value = 0.5, step = 0.1)
            ),
            column(
              6,
              sliderInput("anova.degree", "Degree", min = 1, max = 10, value = 2, step = 1)
            )
          )
          return(f)
        }
      })


      # generate data
      clsdat = reactive({
        if(input$data.type == "Circle") {
          req(input$cls.n, input$cls.nval, input$circle.noise)
          x.train = matrix(rnorm(n = input$cls.n * 2), ncol = 2)
          y.train = 1 * (x.train[, 1]^2 + rnorm(n = input$cls.n, sd = input$circle.noise) + x.train[, 2]^2 > qchisq(p = input$circle.p, df = input$circle.df)) + 1
          x.test = matrix(rnorm(n = input$cls.nval), ncol = 2)
          y.test = 1 * (x.test[, 1]^2 + rnorm(n = input$cls.nval, sd = input$circle.noise) + x.test[, 2]^2 > qchisq(p = input$circle.p, df = input$circle.df)) + 1
          return(list(x.train = x.train,
                      y.train = factor(y.train),
                      x.test = x.test,
                      y.test = factor(y.test)))
        }

        if(input$data.type == "XOR") {
          req(input$cls.n, input$cls.nval, input$xor.noise)
          x.train = matrix(rnorm(n = input$cls.n * 2, mean = 0, sd = input$xor.sigma), ncol = 2)
          y.train = factor(1 * apply(x.train, 1, function(a) {xor(a[1] > rnorm(1, 0, input$xor.noise),
                                                                  a[2] > rnorm(1, 0, input$xor.noise))}) + 1)
          x.test = matrix(rnorm(n = input$cls.nval * 2, mean = 0, sd = input$xor.sigma), ncol = 2)
          y.test = factor(1 * apply(x.test, 1, function(a) {xor(a[1] > rnorm(1, 0, input$xor.noise),
                                                                a[2] > rnorm(1, 0, input$xor.noise))}) + 1)
          return(list(x.train = x.train,
                      y.train = y.train,
                      x.test = x.test,
                      y.test = y.test))
        }

        if(input$data.type == "Gaussian") {
          req(input$gaussian.1.mu1, input$gaussian.1.mu2, input$gaussian.2.mu1, input$gaussian.2.mu2,
              input$gaussian.1.sigma1, input$gaussian.1.sigma2, input$gaussian.1.rho,
              input$gaussian.2.sigma1, input$gaussian.2.sigma2, input$gaussian.2.rho,
              input$cls.n, input$cls.nval)
          mu1 = c(input$gaussian.1.mu1, input$gaussian.1.mu2)
          mu2 = c(input$gaussian.2.mu1, input$gaussian.2.mu2)
          covmat1 = matrix(c(input$gaussian.1.sigma1^2,
                             input$gaussian.1.sigma1 * input$gaussian.1.sigma2 * input$gaussian.1.rho,
                             input$gaussian.1.sigma1 * input$gaussian.1.sigma2 * input$gaussian.1.rho,
                             input$gaussian.1.sigma2^2),
                           nrow = 2)
          covmat2 = matrix(c(input$gaussian.2.sigma1^2,
                             input$gaussian.2.sigma1 * input$gaussian.2.sigma2 * input$gaussian.2.rho,
                             input$gaussian.2.sigma1 * input$gaussian.2.sigma2 * input$gaussian.2.rho,
                             input$gaussian.2.sigma2^2),
                           nrow = 2)
          x.train = rbind(rmvnorm(n = ceiling(input$cls.n / 2), mean = mu1, sigma = covmat1),
                          rmvnorm(n = floor(input$cls.n / 2), mean = mu2, sigma = covmat2))
          y.train = factor(c(rep(1, ceiling(input$cls.n / 2)), rep(2, floor(input$cls.n / 2))))
          x.test = rbind(rmvnorm(n = ceiling(input$cls.nval / 2), mean = mu1, sigma = covmat1),
                         rmvnorm(n = floor(input$cls.nval / 2), mean = mu2, sigma = covmat2))
          y.test = factor(c(rep(1, ceiling(input$cls.nval / 2)), rep(2, floor(input$cls.nval / 2))))
          return(list(x.train = x.train,
                      y.train = y.train,
                      x.test = x.test,
                      y.test = y.test))
        }

        if(input$data.type == "Spiral") {
          req(input$spiral.r, input$spiral.theta1, input$spiral.theta2,
              input$cls.n, input$cls.nval, input$spiral.noise)
          # train
          r1 = seq(from = 0.05, to = input$spiral.r, length.out = ceiling(input$cls.n / 2))
          t1 = seq(from = input$spiral.theta1[1], to = input$spiral.theta1[2], length.out = ceiling(input$cls.n / 2)) +
            rnorm(n = ceiling(input$cls.n / 2), sd = input$spiral.noise)
          r2 = seq(from = 0.05, to = input$spiral.r, length.out = floor(input$cls.n / 2))
          t2 = seq(from = input$spiral.theta2[1], to = input$spiral.theta2[2], length.out = floor(input$cls.n / 2)) +
            rnorm(n = floor(input$cls.n / 2), sd = input$spiral.noise)
          x.train = rbind(
            cbind(r1 * cos(t1), r1 * sin(t1)),
            cbind(r2 * cos(t2), r2 * sin(t2))
          )
          y.train = factor(c(rep(1, ceiling(input$cls.n / 2)), rep(2, floor(input$cls.n / 2))))
          # test
          r1 = seq(from = 0.05, to = input$spiral.r, length.out = ceiling(input$cls.nval / 2))
          t1 = seq(from = input$spiral.theta1[1], to = input$spiral.theta1[2], length.out = ceiling(input$cls.nval / 2)) +
            rnorm(n = ceiling(input$cls.nval / 2), sd = input$spiral.noise)
          r2 = seq(from = 0.05, to = input$spiral.r, length.out = floor(input$cls.nval / 2))
          t2 = seq(from = input$spiral.theta2[1], to = input$spiral.theta2[2], length.out = floor(input$cls.nval / 2)) +
            rnorm(n = floor(input$cls.nval / 2), sd = input$spiral.noise)
          x.test = rbind(
            cbind(r1 * cos(t1), r1 * sin(t1)),
            cbind(r2 * cos(t2), r2 * sin(t2))
          )
          y.test = factor(c(rep(1, ceiling(input$cls.nval / 2)), rep(2, floor(input$cls.nval / 2))))
          return(list(x.train = x.train,
                      y.train = y.train,
                      x.test = x.test,
                      y.test = y.test))
        }
      })


      # fit
      clasfit = reactive({
        x.train = clsdat()$x.train
        y.train = clsdat()$y.train
        if(input$kernel == "RBF") {
          req(input$rbf.sigma)
          kernelname = "rbfdot"
          kpar = list(sigma = input$rbf.sigma)
        }
        else if(input$kernel == "Polynomial") {
          req(input$poly.degree, input$poly.offset, input$poly.scale)
          kernelname = "polydot"
          kpar = list(degree = input$poly.degree, offset = input$poly.offset, scale = input$poly.scale)
        }
        else if(input$kernel == "Linear") {
          kernelname = "vanilladot"
          kpar = list()
        }
        else if(input$kernel == "Hyperbolic tangent") {
          req(input$hyperbolic.scale, input$hyperbolic.offset)
          kernelname = "tanhdot"
          kpar = list(scale = input$hyperbolic.scale, offset = input$hyperbolic.offset)
        }
        else if(input$kernel == "Laplacian") {
          req(input$rbf.sigma)
          kernelname = "laplacedot"
          kpar = list(sigma = input$rbf.sigma)
        }
        else if(input$kernel == "Bessel") {
          req(input$bessel.sigma, input$bessel.degree, input$bessel.order)
          kernelname = "besseldot"
          kpar = list(sigma = input$bessel.sigma, degree = input$bessel.degree, order = input$bessel.order)
        }
        else if(input$kernel == "ANOVA RBF") {
          req(input$anova.sigma, input$anova.degree)
          kernelname = "anovadot"
          kpar = list(sigma = input$anova.sigma, degree = input$anova.degree)
        }
        else {
          kernelname = "splinedot"
          kpar = list()
        }
        req(input$cross, input$c)
        fit = ksvm(x = x.train, y = y.train, C = input$c, cross = input$cross,
                   kernel = kernelname, kpar = kpar)
        return(fit)
      })


      # clasplot
      output$clasplot = renderPlot({
        req(clasfit())
        x.train = clsdat()$x.train
        y.train = clsdat()$y.train
        train = tibble(X1 = x.train[,1], X2 = x.train[,2], y = factor(y.train))
        # grid
        grange = apply(x.train, 2, range)
        x1 = seq(from = grange[1,1], to = grange[2,1], length = 100)
        x2 = seq(from = grange[1,2], to = grange[2,2], length = 100)
        xgrid = expand.grid(X1 = x1, X2 = x2)
        ypredgrid = predict(clasfit(), as.matrix(xgrid))
        df_grid = data.frame(xgrid = xgrid, ypredgrid = ypredgrid)
        ggplot(data = df_grid) +
          geom_tile(aes_(x = ~ xgrid.X1, y = ~ xgrid.X2, fill = ~ ypredgrid), alpha = 0.2) +
          geom_point(data = train, aes_(x = ~ X1, y = ~ X2, color = ~ as.factor(y))) +
          labs(x = "X1", y = "X2", color = "Label (Y)") +
          guides(fill = "none") +
          theme_bw()
      })


      # summary
      output$classum = renderUI({
        req(clasfit())
        # test error
        x.test = clsdat()$x.test
        y.test = clsdat()$y.test
        y.pred = predict(clasfit(), x.test)
        test_error = mean(y.test != y.pred)

        l1 = p(paste0("Objective Function Value: ", clasfit()@obj %>% round(4)))
        l2 = br()
        l3 = p(paste0("Training error: ", clasfit()@error %>% round(4)))
        l4 = br()
        l5 = p(paste0("Cross validation error: ", clasfit()@cross %>% round(4)))
        l6 = br()
        l7 = p(paste0("Test error: ", test_error %>% round(4)))
        l8 = br()
        l9 = p(paste0("Number of Support Vectors: ", clasfit()@nSV))
        return(list(l1, l2, l3, l4, l5, l6, l7, l8, l9))
      })
    }
  )
}
