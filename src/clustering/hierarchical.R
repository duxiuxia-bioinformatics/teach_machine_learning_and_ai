# Author: Xiuxia Du
# Date: Fall 2015, Fall 2016




rm(list=ls())
graphics.off()





par_default <- par()
par(pin=c(14,9))
par(mar=c(1,1,1,7))




# -------------------------------------------------------------
# 1. Import data
# -------------------------------------------------------------
#in_file_name <- "SCLC_study_output.csv"
in_file_name <- "/Users/xdu4/Dropbox (UNC Charlotte)/Duxiuxia/Ocean/Teaching/ML/2016_Fall/Lectures/R/SCLC_study_output.csv"
data_in <- read.csv(file=in_file_name,
                    header=T,
                    check.names = F)



# rows are peaks, columns are values associated with each sample
all_col_names <- colnames(data_in)
# remove dots in column names with a white space
all_col_names <- gsub(pattern="\\.", replacement=" ", x=all_col_names, perl=T)
colnames(data_in) <- all_col_names
variable_names <- data_in$`row ID`



# extract the peak area columns
tf <- grepl(pattern="NSCLC_", x=all_col_names) & grepl(pattern="area", x=all_col_names)
II <- which(tf==T)

tf <- grepl(pattern="^SCLC_", x=all_col_names) & grepl(pattern="area", x=all_col_names)
JJ <- which(tf==T)



# get the peak area data
data <- cbind(data_in[, II], data_in[, JJ])
sample_names <- colnames(data)



crop_name <- function(s) {
  ind <- regexpr(pattern="_POS", text=s)
  return(substr(x=s, start=1, stop=ind-1))
}

sample_names_cropped <- sapply(sample_names, crop_name)
colnames(data) <- sample_names_cropped




# -------------------------------------------------------------
# 2. Filter variables
# -------------------------------------------------------------
tf <- grepl(pattern="row number of detected peaks", x=all_col_names)
II <- which(tf==T)


JJ <- which(data_in[, II] >= 40) # select variables that are detected in all of the samples
data_for_analysis <- data[JJ, ]
data_for_analysis <- as.data.frame(t(data_for_analysis))
colnames(data_for_analysis) <- variable_names[JJ]


write.csv(data_for_analysis, "SCLC_study_output_filtered_2.csv")




# -------------------------------------------------------------
# 2. hierarchical clustering
# -------------------------------------------------------------
# euclidean dissimilarity
distMatrix <- dist(x=data_for_analysis, method="euclidean")


# complete linkage
re.hclust <- hclust(d=distMatrix, method="complete")
cutoff <- 2.5e+6

# vertical dendrogram
plot(re.hclust, main="dist = euclidean, linkage = complete", sub="",
     xlab="", ylab="dissimilarity", 
     hang=-1)
points(c(0, nrow(data_for_analysis)), c(cutoff, cutoff), 
       type="l", lwd=2, lty=2,
       col="red")

# horizontal dendrogram
plot(as.dendrogram(re.hclust), horiz=T, 
     main="dist = euclidean, linkage = complete", 
     xlab="dissimilarity", ylab="")
points(c(cutoff, cutoff), c(0, nrow(data_for_analysis)), 
       type="l", lwd=2, lty=2,
       col="red")



re.hclust$order

# first merge
re.hclust$merge[1,]
re.hclust$labels[c(27,28)]
merge_dist <- re.hclust$height[1]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")

merge_dist
sqrt(sum((data_for_analysis[27,]-data_for_analysis[28,])^2))


# second merge
re.hclust$merge[2,]
re.hclust$labels[c(25,26)]
merge_dist <- re.hclust$height[2]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")

merge_dist
sqrt(sum((data_for_analysis[25,]-data_for_analysis[26,])^2))


# third merge
re.hclust$merge[3,]
re.hclust$labels[c(18,19)]
merge_dist <- re.hclust$height[3]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")

merge_dist
sqrt(sum((data_for_analysis[18,]-data_for_analysis[19,])^2))


# fourth merge
re.hclust$merge[4,]
re.hclust$labels[c(5,8)]
merge_dist <- re.hclust$height[4]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")


# fifth merge
re.hclust$merge[5,]
re.hclust$labels[c(23,24)]
merge_dist <- re.hclust$height[5]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")


# sixth merge
re.hclust$merge[6,]
re.hclust$labels[c(6, 7)]
merge_dist <- re.hclust$height[6]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")


# seventh merge
re.hclust$merge[7,]
re.hclust$labels[c(29, 31)]
merge_dist <- re.hclust$height[7]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")


# eighth merge
re.hclust$merge[8,]
re.hclust$labels[c(21, 22)]
merge_dist <- re.hclust$height[8]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")


# ninth merge
re.hclust$merge[9,]
re.hclust$labels[c(2, 3)]
merge_dist <- re.hclust$height[9]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="green")


# tenth merge
re.hclust$merge[10,]
re.hclust$labels[35]
re.hclust$merge[1,]
re.hclust$labels[c(27, 28)]
merge_dist <- re.hclust$height[10]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="magenta")


# eleventh merge
re.hclust$merge[11,]
re.hclust$labels[c(1, 20)]
merge_dist <- re.hclust$height[11]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="magenta")


# twelfth merge
re.hclust$merge[12,]
re.hclust$labels[c(10, 16)]
merge_dist <- re.hclust$height[12]
points(c(merge_dist, merge_dist), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="magenta")








# single linkage
re.hclust <- hclust(d=distMatrix, method="single")
cutoff <- 1.05e+6


# vertical dendrogram
plot(re.hclust, main="dist = euclidean, linkage = single", sub="",
     xlab="", ylab="dissimilarity",  
     hang=-1)

points(c(0, nrow(data_for_analysis)), c(cutoff, cutoff), type="l", col="red", lwd=2, lty=2)

# horizontal dendrogram
plot(as.dendrogram(re.hclust), horiz=T, 
     main="dist = euclidean, linkage = single", 
     ylab="", xlab="dissimilarity")
points(c(cutoff, cutoff), c(0, nrow(data_for_analysis)), 
       type="l", lwd=2, lty=2,
       col="red")





# average linkage
re.hclust <- hclust(d=distMatrix, method="average")
cutoff <- 1.75e+6

# vertical dendrogram
plot(re.hclust, 
     main="dist = euclidean, linkage = average", sub="", 
     xlab="", ylab="dissimilarity",
     hang=-1)
points(c(0, nrow(data_for_analysis)), c(cutoff, cutoff), type="l", col="red", lwd=2, lty=2)


# horizontal dendrogram
plot(as.dendrogram(re.hclust), horiz=T, 
     main="dist = euclidean, linkage = average", 
     ylab="", xlab="dissimilarity")
points(c(cutoff, cutoff), c(0, nrow(data_for_analysis)), 
       type="l", lwd=2, lty=2,
       col="red")





# centroid linkage
re.hclust <- hclust(d=distMatrix, method="centroid")
cutoff <- 8e+5

# vertical dendrogram
plot(re.hclust, main="dist = euclidean, linkage = centroid", sub="",
     xlab="", ylab="dissimilarity",
     hang=-1)
points(c(0, nrow(data_for_analysis)), c(cutoff, cutoff), 
       type="l", lwd=2, lty=2,
       col="red")

# horizontal dendrogram
plot(as.dendrogram(re.hclust), horiz=T, 
     main="dist = euclidean, linkage = centroid", 
     ylab="", xlab="disimilarity")
points(c(cutoff, cutoff), c(0, nrow(data_for_analysis)), 
       type="l", lwd=2, lty=2,
       col="red")





# manhattan dissimilarity
distMatrix <- dist(x=data_for_analysis, method="manhattan")
cutoff <- 8e+6

# complete linkage
re.hclust <- hclust(d=distMatrix, method="complete")
plot(as.dendrogram(re.hclust), horiz=T,
     main="dist = manhattan, linkage = complete",
     ylab="", xlab="dissimilarity")
points(c(cutoff, cutoff), c(0, nrow(data_for_analysis)),
       type="l", lwd=2, lty=2,
       col="red")


# single linkage
re.hclust <- hclust(d=distMatrix, method="single")

plot(as.dendrogram(re.hclust), horiz=T,
     main="dist = manhattan, linkage = single",
     ylab="", xlab="dissimilarity")



# average linkage
re.hclust <- hclust(d=distMatrix, method="average")

plot(as.dendrogram(re.hclust), horiz=T,
     main="dist = manhattan, linkage = average",
     ylab="", xlab="dissimilarity")




# centroid linkage
re.hclust <- hclust(d=distMatrix, method="centroid")

plot(as.dendrogram(re.hclust), horiz=T,
     main="dist = manhattan, linkage = centroid",
     ylab="", xlab="dissimilarity")


