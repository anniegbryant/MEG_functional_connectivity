# install_r_packages_for_visualization.R
# This script installs the required R packages for visualization of the results.

# Packages on CRAN
packages <- c("broom", "cowplot", "ggpubr", "ggseg", "ggsegSchaefer", "ggsignif", "ggVennDiagram", "glue", "patchwork", "rlist", "see", "tidyverse")



# Install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos="https://cloud.r-project.org/")
  }
}

sapply(packages, install_if_missing)
