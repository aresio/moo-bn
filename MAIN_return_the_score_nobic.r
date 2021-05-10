# set the working directory to be desktop/experiments_nobile
# setwd("~/Desktop/experiments_nobile")
setwd(".")

args <- commandArgs(trailingOnly = TRUE)

# source the needed scripts
source("get.likelihood.score.R")

# read from a file located in the same directory the number of dataset to be used
# curr_dataset = as.numeric(read.table("curr_dataset.txt"))
curr_dataset = as.numeric(as.numeric(args[2]))
curr_dataset = as.matrix(read.table(paste0(getwd(),"/dataset/file_",curr_dataset,".txt"),header=FALSE,sep=",",stringsAsFactors=FALSE))
colnames(curr_dataset) = as.character(1:ncol(curr_dataset))
rownames(curr_dataset) = as.character(1:nrow(curr_dataset))

# read the solution to be evaluated from a file in the same directory named curr_solution.txt
# curr_solution = read.table(paste0(getwd(),"/curr_solution.txt"),header=FALSE,stringsAsFactors=FALSE)
curr_solution = as.matrix(read.table(paste0(getwd(),paste0("/",args[1])),header=FALSE,stringsAsFactors=FALSE))
colnames(curr_solution) = as.character(1:ncol(curr_solution))
rownames(curr_solution) = as.character(1:nrow(curr_solution))

#print(curr_dataset)
#print(curr_solution)

# compute the likelihood BIC score
curr_score_1 = get.likelihood.score(dataset=curr_dataset,adj.matrix=curr_solution,regularizator="loglik")
curr_score_2 = get.likelihood.score(dataset=curr_dataset,adj.matrix=curr_solution,regularizator="bic")
curr_score_2 = curr_score_1-curr_score_2

# print the score to command
cat(curr_score_1,"\n")
cat(curr_score_2,"\n")
