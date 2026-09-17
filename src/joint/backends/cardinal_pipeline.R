args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("usage: cardinal_pipeline.R INPUT_IMZML OUTPUT_DIR TOLERANCE UNIT SNR")
}
if (!requireNamespace("Cardinal", quietly = TRUE)) {
  stop("The Cardinal R package is required")
}
input_imzml <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = FALSE)
tolerance <- as.numeric(args[[3]])
unit <- args[[4]]
snr <- as.numeric(args[[5]])
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
msi <- Cardinal::readMSIData(input_imzml)
processed <- Cardinal::peakProcess(msi, SNR = snr)
aligned <- Cardinal::peakAlign(processed, tolerance = tolerance, units = unit)
Cardinal::writeMSIData(aligned, file = file.path(output_dir, "cardinal.imzML"))
