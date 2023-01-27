library(keras)
library(ggplot2)
library(gganimate)
library(gifski)

setwd("~/Desktop/Baylor/Interns/keras_demo/autoencoder")

## Scale Data
df <- read.csv("../data.csv")
x_train <- df[, -c(1, 2, 33)]
labels <- df$diagnosis
ifelse(labels == "M", "Malignant", "Benign")
x_scaled <- scale(X_train)

input_size <- dim(X_scaled)[2]
latent_size <- 2

enc_input = layer_input(shape = input_size)
enc_output = enc_input %>% 
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = latent_size, activation = "relu")

encoder = keras_model(enc_input, enc_output)

dec_input = layer_input(shape = latent_size)
dec_output = dec_input %>% 
  layer_dense(units = 10, activation = "relu") %>% 
  layer_dense(units = input_size, activation = "linear") 

decoder = keras_model(dec_input, dec_output)

aen_output = enc_input %>% 
  encoder() %>% 
  decoder()

aen = keras_model(enc_input, aen_output)
summary(aen)

aen %>% compile(optimizer = "adam", loss = "mse")

# aen %>% fit(x_scaled, x_scaled, epochs = 100, batch_size = 32, shuffle = TRUE,
#             validation_split = 0.5)

embed <- list()
global_loss <- list()
for (i in 1:100){
  global_loss[[i]] <- aen %>% fit(x_scaled, x_scaled, epochs = 1, batch_size=16,
              validation_split = 0.2)
  
  embed[[i]] <- data.frame(predict(encoder, X_scaled), labels, i)
}

x_decoded <- predict(aen, x_scaled)

par(mfrow = c(1, 2))
boxplot(X_scaled[, 1:5])
boxplot(X_decoded[, 1:5])

plot_df <- do.call(rbind, embed)
colnames(plot_df)[4] <- "epoch"
plot_df$epoch <- as.numeric(plot_df$epoch)

p <- ggplot(plot_df, aes(X1, X2, color = labels)) +
  geom_point() + labs(title = "Epoch: {closest_state}")+
  transition_states(epoch) + theme_bw() +
  enter_fade() + 
  exit_shrink() +
  ease_aes('sine-in-out')
animate(p, renderer = gifski_renderer())
anim_save("file.gif", p)


### Task for D:: Exploration
# Explore various Keras Loss Functions, Optimizers and activation functions and
# Save the embedding gifs to explore how it affects the latent space

