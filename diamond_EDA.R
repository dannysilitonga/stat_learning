library(ggplot2)
library(dplyr)

diamonds <- read.csv("/Users/dannysilitonga/Dropbox/courses/Stat_Learning/projects/data/diamond_prices.csv", 
                     header=TRUE)
diamonds$cut <- factor(diamonds$cut, ordered=TRUE, levels=c("Fair", "Good", "Very Good", "Premium", "Ideal"))
diamonds$color <- factor(diamonds$color, ordered=TRUE, levels=c("J", "I", "H", "G", "F", "E", "D"))
diamonds$clarity <- factor(diamonds$clarity, ordered=TRUE, levels=c("I1", "SI2", "SI1", "VS2", "VS1", "VVS2",
                                                                    "VVS1", "IF")) 
                                                                    
str(diamonds)

# exploring the price variable 
ggplot(diamonds, aes(price)) + 
  geom_histogram(binwidth = 500)

diamonds %>% 
  summarize(mean = mean(price), sd = sd(price), median =median(price))


diamonds %>% 
  group_by(cut) %>%
  summarize(counts = n())

diamonds %>%
  group_by(cut) %>%
  summarise(counts = n() / nrow(diamonds))

diamonds %>%
  group_by(color) %>%
  summarise(counts = n() / nrow(diamonds))

diamonds %>%
  group_by(clarity) %>%
  summarise(counts = n() / nrow(diamonds))


ggplot(data = diamonds, aes(x = color)) +
  geom_bar()

ggplot(data = diamonds, aes(x = clarity)) +
  geom_bar()

ggplot(data = diamonds, aes(x = depth, fill = cut)) +
  geom_histogram(binwidth = 0.2)

ggplot(data = diamonds, aes(x = depth)) +
  geom_histogram(binwidth = 0.2) +
  facet_wrap(~ cut)

ggplot(data = diamonds, aes(x = color)) +
  geom_bar(aes(fill=cut), position = position_stack(reverse=TRUE))

fair_diamonds <- diamonds %>%
  filter(cut == "Fair")
ggplot(data = fair_diamonds, aes(x = price, y = carat)) +
  geom_point()

ggplot(data = diamonds, aes(x = price, y = carat)) +
  geom_point(position = "jitter", alpha=0.05)

#g_diamonds <- diamonds %>%
#  filter(color == "G")
#ggplot(data = g_diamonds, aes(x = price, y = color)) +
#  geom_point(position = "jitter",alpha=0.05)

ggplot(data = diamonds, aes(x = price, y = color)) +
  geom_point(position = "jitter", alpha=0.05)

ggplot(data = diamonds, aes(x = price, y = cut)) +
  geom_point(position = "jitter", alpha=0.05)

ggplot(data = diamonds, aes(x = price, y = clarity)) +
  geom_point(position = "jitter", alpha=0.05)




ggplot(data = diamonds, aes(x = carat, y = price)) +
  geom_point()
ggplot(data = diamonds, aes(x = carat, y = price)) +
  geom_point() +
  ylim(0, 2000) +
  xlim(0, 1)