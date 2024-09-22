library(cowplot)
library(tidyverse)
library(ggseg)
library(ggsegGlasser)
library(patchwork)
library(viridis)
library(fsbrain)
library(glue)

# Set cowplot theme
theme_set(theme_cowplot())

# Define directories
subjects_dir = "~/data/fs/"
glasser_atlas_dir = "~/data/neuroimaging_atlases/surfaces/Glasser_2016/fsaverage/"
github_dir <- "~/github/MEG_functional_connectivity/"

# fsaverage surface
subject_id = 'fsaverage'

# Glasser left/right surface annot
lh_glasser_annot = freesurferformats::read.fs.annot(glue("{glasser_atlas_dir}/lh.HCP-MMP1.annot"))
rh_glasser_annot = freesurferformats::read.fs.annot(glue("{glasser_atlas_dir}/rh.HCP-MMP1.annot"))

# 1 = "V1", "V2"
# 2 = "LO1", "LO2", "LO3"
# 3 = "FFC"
# 4 = "IPS1"
# 5 = "46", "9-46d", "a9-46d", "a9-46v"
lh_glasser_parcels_of_interest <- case_when(lh_glasser_annot$label_names %in% c("L_V1_ROI", "L_V2_ROI") ~ 1,
                                            lh_glasser_annot$label_names %in% c("L_LO1_ROI", "L_LO2_ROI", "L_LO3_ROI") ~ 2,
                                            lh_glasser_annot$label_names %in% c("L_FFC_ROI") ~ 3,
                                            lh_glasser_annot$label_names %in% c("L_IPS1_ROI") ~ 4,
                                            lh_glasser_annot$label_names %in% c("L_46_ROI", "L_9-46d_ROI", "L_a9-46d_ROI", "L_a9-46v_ROI") ~ 5,
                                            T ~ NA_real_)
rh_glasser_parcels_of_interest <- case_when(rh_glasser_annot$label_names %in% c("R_V1_ROI", "R_V2_ROI") ~ 1,
                                            rh_glasser_annot$label_names %in% c("R_LO1_ROI", "R_LO2_ROI", "R_LO3_ROI") ~ 2,
                                            rh_glasser_annot$label_names %in% c("R_FFC_ROI") ~ 3,
                                            rh_glasser_annot$label_names %in% c("R_IPS1_ROI") ~ 4,
                                            rh_glasser_annot$label_names %in% c("R_46_ROI", "R_9-46d_ROI", "R_a9-46d_ROI", "R_a9-46v_ROI") ~ 5,
                                            T ~ NA_real_)

glasser_atlas_rois_on_surface = vis.symmetric.data.on.subject(subjects_dir, vis_subject_id=subject_id, 
                                                              morph_data_lh=lh_glasser_parcels_of_interest, 
                                                              morph_data_rh=rh_glasser_parcels_of_interest, 
                                                              bg="sulc_light",
                                                              makecmap_options = list('colFn'=viridis),
                                                              surface="inflated", draw_colorbar = T,
                                                              rglactions = list('shift_hemis_apart'=TRUE, 'no_vis'=T))


# Export the vis
export(glasser_atlas_rois_on_surface, img_only = TRUE, 
       output_img = glue("{github_dir}/plots/Glasser_atlas_ROIs.png"),
       rglactions = list('shift_hemis_apart'=TRUE,'no_vis'=T), 
       view_angles=fsbrain::get.view.angle.names(angle_set = "t9"))