# ---
# title: "What drives the sharing of Mashable Articles?"
# author:
# - Dylan Chou
# - dvchou
# date: "null"
# output:
#   pdf_document:
#     toc: no
#   word_document:
#     toc: no
# ---

#```{r, include=FALSE}
###########################
# STYLE EDITS: IGNORE THIS
###########################
knitr::opts_chunk$set(message = FALSE) # include this if you don't want markdown to knit messages
knitr::opts_chunk$set(warning = FALSE) # include this if you don't want markdown to knit warnings
knitr::opts_chunk$set(echo = TRUE) # set echo=FALSE to hide code from html output
#```

#```{r,echo=FALSE}
library("knitr")
library("cmu202")
library("kableExtra")
library("pander")
library("readr")
library("magrittr")
library("car")
library("dplyr")
#```

#```{r,echo=FALSE}
social <- read.csv("social_project02.csv", header = TRUE)
#```

# Introduction

#With the sweeping wave of new technologies and increasingly interconnected online networks, sharing has become easier than ever before. However, with the large volume of users on social media, it's important to determine what drives people to share the things they do. We will analyze the topic of sharing social media articles as we intend to learn more about the reasons why people spread certain information. Determining the driving forces that motivate people to share certain things such as articles can be imperative in funneling out the massive amounts of data in traffic of shared content through the Internet and social media and analyzing the important facets of an article or shareable content that would encourage someone to share it and allowing a digital company, such as Mashable, to discover factors that can perhaps boost their profit with more people sharing or reading their articles. If Mashable finds out factors that help the number of shares of their articles, the digitial company could provide articles that are strong in those factors so more people will read and share them, leading to greater success. To mine the data and determine what variables predict the amount of shares an article gets, data samples are taken from the digital media website Mashable with only the key variables of interest included in a spreadsheet. Given the information from Mashable articles gathered for over 2 years, we intend to answer the question of what variables can predict the amount of shares an article obtains. 

# Exploratory Data Analysis and Data Cleaning

# After obtaining the spreadsheet of data about the social project, we will first familiarize ourselves with what is contained in the dataset. We can look at the first and last several rows of data regarding social media sharing and move onto univariate data visualization and analysis. 

## Observing Snippets of Data 

#It's important to note that the data comprises 4351 observations or articles with 5 variables. The days published variable is one of the 7 days in the week, sentiment being either positive or negative, channel being Business, Technology, Entertainment, World and Other, and content being categorized later in the analysis of categorical predictor variables. Regarding our variables of interest, we intend to predict the number of shares of a Mashable article from the variables of content, the day it was published, the overall positive or negative tone in the article, and the channel or type of website. 

#| Variable | Description | 
#|---------------|-------------------------------------------------:|
#|**shares**     | The number of article shares among sample of Mashable articles|
# |**content**    | The word count in the article|
# |**daypublished**     | The day during the week when the article was published|
# |**sentiment**| The overall tone or sentiment, positivity or negativity, in the article| 
# |**channel**| The type of website or topic of the article (Business, Tech, (etc.)) | 

# To become familiar with the data, we will look at the first rows in the social project dataset: 

# ```{r,echo=FALSE}
par(mfrow = c(1,2))
print("The First Six Rows of Mashable Article Dataset:")
print(head(social))
print("The Last Rows of Mashable Article Dataset")
print(tail(social))
# ```

# ```{r,echo=FALSE}
colMax <- function(data) sapply(data, max, na.rm = TRUE)
colMin <- function(data) sapply(data, min, na.rm = TRUE)
print("Maximum Values For Number of Shares and Article Content")
colMax(social[c(1:2)])
print("Minimum Values For Number of Shares and Article Content")
colMin(social[c(1:2)])
# ```

# The magnitude of the shares are notably high and roughly in the high hundreds or thousands. The content, or word count, in the article appears to vary greatly from low hundreds all the way to the thousands. The sentiments and days published appear to be as expected where daypublished are the days of the week and the sentiment could either be defined as positive or negative. Notably, the maximum number of shares was 8900 shares with the lowest number of shares of an article being 22 shares. The largest word count was 2365 words and the lowest was 0 words. The other variables were categorical.

## Univariate Exploratory Data Analysis 

# After observing the magnitude of the data variables, we will move onto observing the distribution of the response variable and explanatory variables in histograms, boxplots, barcharts as well as with summary statistics. It's important to note that for the ANOVA model that we will ultimately perform on the data, the content or word count in the articles has been categorized as $1: 0 \ words \leq content < 200 \ words, \ 2:200 \ words \leq content < 400 \ words, \ 3: 400 \ words \leq content < 600 \ words, \ 4: 600 \ words \leq content < 800 \ words, \ 5: 800 \ words \leq content < 1000 \ words,\ 6: 1000 \ words \leq content < 1200 \ words, \ 7: 1200 \ words \leq content < 1400 \ words,\ 8: 1400 \ words \leq content < 1600 \ words,\ 9: 1600 \ words \leq content < 1800 \ words,\ 10: 1800 \ words \leq content < 2000 \ words,\ 11: 2000 \ words \leq content < 2200 \ words, \ 12: 2200 \ words \leq content < 2400 \ words$. The content was divided into 4 categories because given that the largest value in the data set was 2365 and we were to categorize the word count by even numbers, 2400 can be divided by 12 to capture ranges of values in increments of 200 words. The choice for the categorization of content into 12 groups of 200 word increments was from the histogram of the quantity variable content, which yielded roughly 12 bins. 

#```{r,echo=FALSE}
words.vect <- rep(0,nrow(social))
social$WordContent <- words.vect
social$WordContent <- ifelse(social$content >= 2200,
                          "12",
                          ifelse(social$content >= 2000, 
                                 "11",
                                 ifelse(social$content >= 1800, 
                                        "10",
                                        ifelse(social$content >= 1600,
                                               "9",
                                               ifelse(social$content >= 1400, 
                                                      "8", 
                                                      ifelse(social$content >= 1200, 
                                                             "7", 
                                                             ifelse(social$content >= 1000,
                                                                    "6",
                                                                    ifelse(social$content >= 800, 
                                                                           "5",
                                                                           ifelse(social$content >= 600, 
                                                                                  "4", 
                                                                                  ifelse(social$content >= 400, 
                                                                                         "3",
                                                                                         ifelse(social$content>=200, 
                                                                                                "2","1")))))))))))
#```

# We will first observe the distribution of the quantitative response variable of the number of Mashable article shares. 

#```{r,echo=FALSE}
par(mfrow = c(1,2))
hist(social$shares, xlab = "Number of Mashable Article Shares ", main = "Histogram of Mashable Article Shares", xlim = c(0,10000),cex.lab=0.8,cex.main=0.8)
boxplot(social$shares,ylab="Number of Mashable Article Shares", main = "Boxplot of Mashable Article Shares",cex.lab=0.8,cex.main=0.8)
summary(social$shares)
#```

# After constructing the histogram of the Mashable article shares, we observe a strongly right-skewed and unimodal distribution in the distribution of the number of Mashable article **shares**, shown in one mode within the distribution and most of the number of article shares being fairly low. There are many outliers in the number of shares as the number of data points beyond the upper fence of the boxplot, so we would have to transform the shares distribution. The median of number of article shares is at 1300 and the spread of the data is roughly 1336 shares, which is the range of shares that captures the middle 50% of the distribution of article shares. In the data visualizations below, we try transforming the distribution of shares by taking the square root of the number of article shares to improve the symmetry of the distribution of article shares. 

# ```{r,echo=FALSE}
par(mfrow=c(1,2))
hist(sqrt(social$shares),main="Histogram of Square Root of Number of Shares",xlab="Square Root of Number of Mashable Article Shares",xlim=c(0,100),cex.lab = 0.7,cex.main=0.7)
boxplot(sqrt(social$shares),main="Boxplot of Square Root of Number of Shares",ylab="Square Root of Number of Mashable Article Shares",cex.lab=0.8,cex.main=0.7)
summary(sqrt(social$shares))
# ```

# After having performing other transformations to improve the symmetry of the response variable and reduce outliers, the logarithmic transformation creates too many small outliers and smaller fractional power transformations create both very small and very large outliers. The square root transformation of the number of shares still yields quite some large outliers, it's the most ideal transformation as the histogram appears to considerably more symmetric. The univariate exploratory data analysis reveals that the distribution of the square root of the number of Mashable article shares is unimodal (one peak) and roughly symmetric, although still slightly right skewed from the histogram. There is still a considerable amount of outliers, but it could simply be an innate component of the data. When outliers are not a rarity, we would conclude the skewness and outliers in the data to be attributed to the data itself. The center measure is the median of the square root of the number of article shares is 36.06. The spread of the data can be observed in the range of the data (the minimum number of square root of shares is 4.69 and maximum square root of shares being 94.34) which is 89.65. The interquartile range of the square root of number of Mashable article shares is 17.20. We will continue with our exploratory data analysis by analyzing the categorical predictor variables in tabular and bar chart form.

# ```{r,echo=FALSE}
par(mfrow = c(2,2))
table(social$WordContent)
barplot(table(social$WordContent),cex.name=0.5)
table(social$channel)
barplot(table(social$channel),cex.name = 0.45)
table(social$daypublished)
barplot(table(social$daypublished),cex.name = 0.45)
table(social$sentiment)
barplot(table(social$sentiment))
# ```

# The **categorical content, WordContent, variable** has 709 articles (16.30% of articles) that are between 0 and 200 words (200 exclusive), 1418 articles (32.59% of articles) between 200 and 400 words (400 exclusive), 825 articles (18.96% of articles) between 400 and 600 words (600 exclusive), 564 articles (12.96% of articles) between 600 words and 800 words (800 exclusive), 313 articles (7.19% of articles) between 800 and 1000 words (1000 exclusive), 215 articles (4.94% of all articles) between 1000 and 1200 words (1200 exclusive), 118 articles (2.71% of articles) between 1200 and 1400 words (1400 exclusive), 72 articles (1.65% of articles) between 1400 and 1600 words (1600 exclusive), 43 articles (0.988% of articles) between 1600 and 1800 words (1800 exclusive), 45 articles (1.03% of articles) between 1800 and 2000 words (2000 exclusive), 21 articles (0.48% of articles) between 2000 and 2200 words (2200 exclusive), and 8 articles (0.184% of articles) between 2200 and 2400 articles. Among the Mashable articles, a majority of them have a category 1 content with a word count that is between 0 and 600 words (600 exclusive). The category 12 content contains the fewest articles (8) having between 2200 and 2400 words (2400 exclusive). In general, aside from the transition from the first to second categories, for the content categories with ranges of higher word counts, there is a lower frequency of articles with that many words.

# Among the Mashable articles, regarding their **channel, or type of website**, the channel containing the fewest articles was Lifestyle as 249 articles (5.72% of the articles) covered the topic of Lifestyle. 730 articles (16.78% of the articles) covered the Business channel, 655 articles (15.05% of the articles) covered the Other channel, 857 articles (19.70% of the articles) covered the Entertainment channel, 848 articles (19.49% of the articles) covered the Technology channel, and 1012 articles (23.26% of the articles) covered the World channel. The channel consisting of the most articles was the World channel with 1012 articles, 23.26% of all the articles.

# The **daypublished, or the day when an article was published** had Saturday as the day with the fewest articles being published on as 253 articles (5.81% of articles) were published on Saturday. 322 articles (7.40% of the articles) were published on Sunday, 738 articles (16.96% of the articles) were published on Monday, 801 articles (18.41% of the articles) published on Tuesday, 864 articles (19.86% of the articles) published on Wednesday, 760 articles (17.47% of the articles) published on Thursday, and 613 articles (14.09% of the articles) published on Friday. Most of the articles were published on Wednesday (19.86%, nearly 20% of the articles (864 articles) were published on Wednesday).

# The **sentiment** among the articles had an overwhelming majority of positive sentiment. There were 3810 articles (87.57% of the articles) that constituted positive sentiment and 541 articles (12.43% of the articles) with negative sentiment. Many more articles were positive than negative.

## Bivariate Exploratory Data Analysis

# In terms of bivariate data analysis, we will observe the boxplots of each predictor against the response variable of the number of shares Mashable articles got. We will then produce interaction plots between each combination of the two predictors to determine if there's any notable interaction between the categorical predictors.

# ```{r,echo=FALSE}
par(mfrow = c(1,2))
dplyr::summarize(group_by(social, daypublished), 
          mean(sqrt(shares)), sd(sqrt(shares)), n())
dplyr::summarize(group_by(social, sentiment), 
          mean(sqrt(shares)), sd(sqrt(shares)), n())
dplyr::summarize(group_by(social, WordContent), 
          mean(sqrt(shares)), sd(sqrt(shares)), n())
dplyr::summarize(group_by(social, channel), 
          mean(sqrt(shares)), sd(sqrt(shares)), n())
boxplot(sqrt(shares) ~ daypublished, data = social,main="Boxplot of Shares Per Day Published",xlab="Day Article was Published",ylab="Number of Article Shares",cex.axis = 0.3,cex.main=0.7)
boxplot(sqrt(shares) ~ sentiment, data = social,main="Boxplot of Shares Per Sentiment",xlab="Type of Sentiment",ylab="Number of Article Shares",cex.main=0.7)
boxplot(sqrt(shares) ~ WordContent, data = social,main="Boxplot of Shares Per Content Category",xlab="Content Category",ylab="Number of Article Shares",cex.main=0.7)
boxplot(sqrt(shares) ~ channel, data = social,main="Boxplot of Shares Per Channel",xlab="Type of Website, or Channel",ylab="Number of Article Shares",cex.axis=0.3,cex.main=0.7)
# ```

# From summary tables of the side-by-side boxplots and the plots itself for each categorical predictor variable, we observe some relationships. The boxplots all contain outliers, but that remains a characteristic of the number of article share data, after transformed by the square root. In the boxplots of the response variable of the square root of number of article shares plotted against each categorical predictor, every plot contains significant outliers, but that likely has something to do with the  The boxplots for the square root of number of article shares against the type of sentiment of the article reveal that positive articles have slightly greater value of square root of article shares, on average, than negative articles. The spread, standard deviation, in the square root of article shares is essentially the same between positive and negative sentiment (15.1 vs. 15.5) although positive sentiment has slightly greater variability. In the boxplots for categorized content, there is a slight, but insignificant difference in square root of article shares between the content categories (or word count categories) in the articles as the medians all appear to be essentially the same except for slightly higher medians for category group 9 (1600 to 1800 words) and group 12 (2200 to 2400 words). The category with the greatest spread, 16.5, is 8 (1400 to 1600 words) while category 12 (2200 to 2400 words) has the least variability, 14.0. The boxplots for the day an article was published appears to convey a significant difference as the square root of the number of article shares appeared to be higher on average over the weekend than the weekdays. The day with the most variability is on Saturday (16.8) while Friday seems to have the least variability (14.7). Also, there appears to be noticeable differences in the number of article shares between the channels in their boxplots. Namely, articles from the Lifestyle channel appear to be shared the most while the articles from Entertainment and World channels seem to be shared the least. The World channel appears to have the least variability (13.3) while Other channel had the highest variability (17.2). 

# ```{r,echo=FALSE}
interaction.plot(x.factor = social$daypublished, 
                 trace.factor = social$channel, 
                 response = sqrt(social$shares),
                 xlab="Day Article was Published",
                 ylab="Square Root of Number of Mashable Article Shares",
                 trace.label ="Channel, Type of Website",main="Interaction Plot of Square Root of Article Shares vs. Day Published Per Channel",cex.axis=0.6)
interaction.plot(x.factor = social$daypublished, 
                 trace.factor = social$WordContent, 
                 response = sqrt(social$shares),
                 xlab="Day Article was Published",
                 ylab="Square Root of Number of Mashable Article Shares",
                 trace.label ="Content, Word Count in Article",main="Interaction Plot of Square Root of Article Shares vs. Day Published Per Content Category",cex.axis=0.6)
interaction.plot(x.factor = social$daypublished, 
                 trace.factor = social$sentiment, 
                 response = sqrt(social$shares),
                 xlab="Day Article was Published",
                 ylab="Square Root of Number of Mashable Article Shares",
                 trace.label ="Article Sentiment",main="Interaction Plot of Square Root of Article Shares vs. Day Published Per Sentiment",cex.axis=0.6)

interaction.plot(x.factor = social$WordContent, 
                 trace.factor = social$channel, 
                 response = sqrt(social$shares),
                 xlab="Content, Word Count Category",
                 ylab="Square Root of Number of Mashable Article Shares",
                 trace.label ="Article Channel, Type of Website",main="Interaction Plot of Square Root of Article Shares vs. Word Content Per Channel",cex.axis=0.6)

interaction.plot(x.factor = social$WordContent, 
                 trace.factor = social$sentiment, 
                 response = sqrt(social$shares),
                 xlab="Content, Word Count Category",
                 ylab="Square Root of Number of Mashable Article Shares",
                 trace.label ="Article Sentiment",main="Interaction Plot of Square Root of Article Shares vs. Word Content Per Sentiment",cex.axis=0.6)

interaction.plot(x.factor = social$sentiment, 
                 trace.factor = social$channel, 
                 response = sqrt(social$shares),
                 xlab="Article Sentiment",
                 ylab="Square Root of Number of Mashable Article Shares",
                 trace.label ="Article Channel, Type of Website",main="Interaction Plot of Square Root of Article Shares vs. Sentiment Per Channel",cex.axis=0.6)
# ```

# These interaction plots collectively seem to explain individual main effects for the day an article was published, the channel, or type of website, and sentiment, but most often don't reveal interaction effects as there isn't apparent change in trace factor effect over different levels of the x factor. However, the interaction of the square root of articles shares against the word count, or content, categorized per channel (type of website) seems to reveal an interaction effect as Business channel appears to have a different relationship over content categorized than, say the other channel. 

# Modeling

# We will construct an ANOVA model in order to predict article shares from the explanatory variables. Given that there are 4 factors in this ANOVA, we must consider creating a factorial ANOVA model with interaction terms. We acknowledge that there could be interactions be three categorical predictors, but most often they result in very insignificant terms. We run a model of the ANOVA with the possible interaction terms between the categorical predictors, then rerun the model with the significant interaction term as a significant interaction term would suffice for a 4 factor ANOVA interaction model. 

# ```{r,echo=FALSE}
social_anova <- aov(sqrt(shares) ~ factor(WordContent) + factor(channel) + factor(daypublished) + factor(sentiment) + factor(channel):factor(WordContent) + factor(WordContent):factor(daypublished) + factor(WordContent):factor(sentiment) + factor(channel):factor(daypublished)+factor(channel):factor(sentiment)+factor(daypublished):factor(sentiment),data=social)
summary(social_anova)
# ```

# ```{r,echo=FALSE}
final_social_anova <- aov(sqrt(shares) ~ factor(WordContent) + factor(channel) + factor(daypublished) + factor(sentiment) + factor(WordContent):factor(channel),data=social)
summary(final_social_anova)
plot(final_social_anova$fitted.values, final_social_anova$residuals)
abline(h=0)
qqnorm(rstandard(final_social_anova))
qqline(rstandard(final_social_anova))
# ```

# ```{r,echo=FALSE}
coefficients(final_social_anova)
# ```

# Having generated our first ANOVA model with the possible interaction terms in the ANOVA, we will move onto predicting the number of shares based on our ANOVA model from an article with particular traits.

# Prediction

# Having constructed the ANOVA model, we will carry on with the prediction of the number of shares from an article with content word count of 500 words, a positive sentiment, being published on Monday and being a Business channel. Below we determine the coefficients of the regressional equivalent of the anova model produced of the 4 factor ANOVA.

# Given the above arguments in the regression equation, we 

# # Discussion

# After mining the data, analyzing the article data, and constructing a model to predict article shares, we can conclude that articles 


