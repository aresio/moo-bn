# return the score of a network based either on loglik, aic or bic score
"get.likelihood.score" = function( dataset, adj.matrix, regularizator ) {
    
    # load the packages for the likelihood fit
    suppressPackageStartupMessages(require("bnlearn"))
    
    # create a categorical data frame from the dataset
    data = array("missing",c(nrow(dataset),ncol(dataset)));
    colnames(data) = as.character(1:ncol(data))
    rownames(data) = as.character(1:nrow(data))
    for (i in 1:nrow(dataset)) {
        for (j in 1:ncol(dataset)) {
            if(dataset[i,j]==1) {
                data[i,j] = "observed";
            }
        }
    }
    data = as.data.frame(data);
    my.names = names(data);
    for (i in 1:length(my.names)) {
        my.names[i] = toString(i);
    }
    colnames(data) = my.names;
    data[,my.names] = lapply(data[,my.names],as.factor)
    
    # create the network
    new.net = empty.graph(colnames(data))
    for (i in 1:nrow(adj.matrix)) {
        for (j in 1:ncol(adj.matrix)) {
            if(adj.matrix[i,j]==1) {
                new.net = set.arc(new.net, colnames(data)[i], colnames(data)[j])
            }
        }
    }
    
    # compute the score for the network
    net_score = NA
    if(regularizator=="loglik") {
        net_score = logLik(new.net,data)
    }
    else if(regularizator=="aic") {
        net_score = AIC(new.net,data)
    }
    else if(regularizator=="bic") {
        net_score = BIC(new.net,data)
    }
    
    return(net_score);

}
