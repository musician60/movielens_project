############################################################
# Create edx set, validation set (final hold-out test set) #
############################################################

# Note: this process could take a couple of minutes
options(warn = -1)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           #title = as.character(title),
                                           #genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

####################################################
# Use the edx dataset to train and test the models #
####################################################

### partition the data set into train_set and test_set
dat <- edx
set.seed(1, sample.kind ="Rounding")
test_index <- createDataPartition(y = dat$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- dat[-test_index, ]
test_set <- dat[test_index, ]


### examine the genre data
#create a table of ratings for each genre
movie_genres <- c("Action", "Adventure", "Animation", "Children", "Comedy",
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                  "Thriller", "War", "Western")

#the following function returns all the ratings for a specified genre
get_ratings <- function(genre) {
    train_set %>%
    #find all the observations for the specified genre
    filter(str_detect(genres, genre)) %>%
    #replace the genres field with the name of the specified genre
    mutate(genres = genre) %>%
    #return the movie Id, user ID, genre, and ratings for the specified genre
    select(movieId, userId, genres, rating)
}
#create a table of ratings for each genre in movie_genres
ratings_table<-data.frame(movieId = integer(), userId = integer(),
                          genres = character(), rating = numeric())
for (i in 1:length(movie_genres)) {
  result <- get_ratings(movie_genres[i])
  ratings_table <- bind_rows(ratings_table, result)
}

#display the number of observations for each genre
ratings_table %>%
  group_by(genres) %>%
  summarize(frequency = n(), .groups = "drop") %>%
  arrange(desc(frequency))

#plot the frequency of the ratings for each genre
ratings_table %>%
  group_by(genres) %>%
  summarize(num_ratings = n(), .groups = "drop") %>%
  ggplot(aes(reorder(genres, num_ratings),
             num_ratings / 1000000,
             fill = genres))+
  geom_bar(stat = "identity", width = 0.9) +
  theme(legend.position = "none") +
  xlab("")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +   
  ylab("total number of ratings (in millions)")

#plot the distribution of the ratings for each genre
ratings_table %>% 
  mutate(genres = fct_reorder(genres, rating, .fun = 'median' )) %>% 
  ggplot(aes(genres, rating, fill = genres)) +
  geom_boxplot() +
  xlab("")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  theme(legend.position = "none")

#numerical summaries of rating data by genre
ratings_table %>%
  group_by(genres) %>%
  summarize(min = min(rating),
            Q1 = quantile(rating, 0.25),
            median = median(rating),
            mean=mean(rating),
            Q3 = quantile(rating, 0.75),
            max = max(rating), .groups = "drop") %>%
  arrange(median)

#plot the average rating for each genre
avg_rating<-ratings_table %>%
  group_by(genres) %>%
  summarize(avg_rating = mean(rating), .groups = "drop")

overall_average_rating <- mean(ratings_table$rating)

avg_rating %>%
  ggplot(aes(reorder(genres, avg_rating), avg_rating, fill = avg_rating))+
  geom_bar(stat = "identity", width = 0.9) +
  geom_hline(yintercept = overall_average_rating, lwd = 1, color = "black") +
  geom_text(aes(4, overall_average_rating + 0.2,
                label = "overall average rating")) +
  ylim(0, 6)+
  xlab("")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 

#define a function to calculate the RMSE
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings) ^ 2))
}

### model one
# calculate the initial prediction
initial_prediction <- mean(train_set$rating)

#calculate the RMSE for model one
rmse <- RMSE(test_set$rating, initial_prediction)

#create a table to keep track of the results
results <- tibble(method = "model_one", rmse=rmse)
results

### model two - movie bias
#find the average prediction error for each movie
movie_bias <- train_set %>%
  group_by(movieId) %>%
  summarize(mbias = mean(rating - initial_prediction), .groups = "drop")

#plot the distribution of the movie_bias
movie_bias %>% ggplot(aes(mbias)) +
  geom_histogram(bins = 10, fill = "blue") +
  xlim(-3.5, 2) +
  xlab("movie bias")

#calculate predicted ratings using model two
predicted_ratings <- test_set %>%
  select(movieId) %>%
  left_join(movie_bias, by = "movieId") %>%
  mutate(mbias = ifelse(is.na(mbias), mean(mbias, na.rm = TRUE), mbias)) %>%
  summarize(pred = initial_prediction + mbias) %>%
  pull(pred)

#calculate the RMSE for model two
rmse <- RMSE(predicted_ratings, test_set$rating)

#add the results to the results table and compare the results
results <- bind_rows(results,
                     tibble(method="model_two", rmse=rmse))
results

### model three - user bias
#find the average prediction error for each user
user_bias <- train_set %>%
  select(movieId, rating, userId) %>%
  left_join(movie_bias, by = "movieId") %>%
  group_by(userId) %>%
  summarize(ubias = mean(rating - initial_prediction - mbias), .groups = "drop")

#show variability of the user_bias
user_bias %>% ggplot(aes(ubias)) +
  geom_histogram(bins = 10, fill = "red") +
  xlim(-3.5, 2) +
  xlab("user bias")

#calculate predicted ratings for model three
predicted_ratings <- test_set %>%
  select(movieId, userId) %>%
  left_join(movie_bias, by="movieId") %>%
  mutate(mbias = ifelse(is.na(mbias), mean(mbias, na.rm=TRUE), mbias)) %>%
  left_join(user_bias, by = "userId") %>%
  mutate(ubias = ifelse(is.na(ubias), mean(ubias, na.rm=TRUE), ubias))%>%
  summarize(pred = initial_prediction + mbias + ubias) %>%
  pull(pred)

#calculate the RMSE for model three
rmse <- RMSE(predicted_ratings, test_set$rating)

#compare the results
results <- bind_rows(results,
                     tibble(method = "model_three", rmse = rmse))
results

### model four - genre bias
#find the average prediction error for each genre
genre_bias <- ratings_table %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  group_by(movieId) %>%
  summarize(gbias = mean(rating - initial_prediction - mbias - ubias), .groups = "drop")

#show variability of the genre_bias
genre_bias %>% ggplot(aes(gbias)) +
  geom_histogram(bins = 10, fill = "purple") +
  xlim(-3.5, 2) +
  xlab("genre bias")

#determine predicted ratings for model four
predicted_ratings <- test_set %>%
  left_join(movie_bias, by = "movieId") %>%
  mutate(mbias = ifelse(is.na(mbias), mean(mbias, na.rm=TRUE), mbias)) %>%
  left_join(user_bias, by = "userId") %>%
  mutate(ubias = ifelse(is.na(ubias), mean(ubias, na.rm=TRUE), ubias)) %>%
  select(movieId, mbias, ubias) %>%
  left_join(genre_bias, by = "movieId") %>%
  mutate(gbias = ifelse(is.na(gbias), mean(gbias, na.rm = TRUE), gbias)) %>%
  summarize(pred = initial_prediction + mbias +ubias +gbias) %>%
  pull(pred)

#calculate the RMSE for model four
rmse <- RMSE(predicted_ratings, test_set$rating)

#compare the results
results <- bind_rows(results,
                     tibble(method = "model_four", rmse = rmse))
results

##################################################
# Use the validation set to test the final model #
##################################################

predicted_ratings <- validation %>%
  left_join(movie_bias, by = "movieId") %>%
  mutate(mbias = ifelse(is.na(mbias), mean(mbias, na.rm = TRUE), mbias)) %>%
  left_join(user_bias, by = "userId") %>%
  mutate(ubias = ifelse(is.na(ubias), mean(ubias, na.rm = TRUE), ubias)) %>%
  select(movieId, mbias, ubias) %>%
  left_join(genre_bias, by = "movieId") %>%
  mutate(gbias = ifelse(is.na(gbias), mean(gbias, na.rm = TRUE), gbias)) %>%
  summarize(pred = initial_prediction + mbias +ubias +gbias) %>%
  pull(pred)

#Use the RMSE to determine how close our predicted ratings are
#to the true values in the validation set
rmse <- RMSE(predicted_ratings, validation$rating)
rmse