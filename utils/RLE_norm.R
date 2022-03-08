#This function conducts RLE normalization of RNA-seq dataset
RLE_normalization  <- function(df, reads_thresh=10, percentage_thresh=0.5, constant=10^6){
        library(edgeR)
        #A) Remove unexpressed genes (for RNA-seq)
        r <- apply(df, 1, function(x){sum(x>=reads_thresh)})>=ncol(df)*percentage_thresh
        df_filtered <- df[r,]
        
        #Now, we need to get library sizes vector
        lib.size = apply(df_filtered, 2, sum)
        
        #Now we should obtain normalization factors for each samples using RLE technique
        edger.rle = lib.size * calcNormFactors(df_filtered, method="RLE")
        
        #Finally we should divide expression values on normalization factors
        RLE_dataset <- sweep(df_filtered, 2, edger.rle, "/") * constant
        
        return(RLE_dataset)
}