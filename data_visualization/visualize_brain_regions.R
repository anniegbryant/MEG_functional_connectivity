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
github_dir <- "~/github/MEG_functional_connectivity/"

# fsaverage surface
subject_id = 'fsaverage'

############ DESIKAN-KILLIANY ##############
# 1. Category-selective: "lateraloccipital", "fusiform"
# 2. Visual: "pericalcarine", "lingual"
# 3. Inferior parietal sulcus: "superior"
# 4. Prefrontal cortex: "rostralmiddlefrontal"

# Desikan-Killiany left/right surface annot
dk_atlas_dir = "~/data/fs/fsaverage/label/"
lh_dk_annot = freesurferformats::read.fs.annot(glue("{dk_atlas_dir}/lh.aparc.annot"))
rh_dk_annot = freesurferformats::read.fs.annot(glue("{dk_atlas_dir}/rh.aparc.annot"))

lh_dk_parcels_of_interest <- case_when(lh_dk_annot$label_names %in% c("fusiform") ~ 1,
                                       lh_dk_annot$label_names %in% c("pericalcarine", "lingual") ~ 2,
                                       lh_dk_annot$label_names %in% c("superiorparietal") ~ 3,
                                       lh_dk_annot$label_names %in% c("rostralmiddlefrontal") ~ 4,
                                       T ~ NA_real_)
rh_dk_parcels_of_interest <- case_when(rh_dk_annot$label_names %in% c("fusiform") ~ 1,
                                       rh_dk_annot$label_names %in% c("pericalcarine", "lingual") ~ 2,
                                       rh_dk_annot$label_names %in% c("superiorparietal") ~ 3,
                                       rh_dk_annot$label_names %in% c("rostralmiddlefrontal") ~ 4,
                                       T ~ NA_real_)

dk_atlas_rois_on_surface = vis.symmetric.data.on.subject(subjects_dir, vis_subject_id=subject_id, 
                                                         morph_data_lh=lh_dk_parcels_of_interest, 
                                                         morph_data_rh=rh_dk_parcels_of_interest, 
                                                         bg=NULL,
                                                         makecmap_options = list('colFn'=viridis),
                                                         surface="inflated", draw_colorbar = T,
                                                         rglactions = list('shift_hemis_apart'=TRUE, 
                                                                           'no_vis'=T,
                                                                           'light_intensity' = 0.2,  # Light intensity adjustment
                                                                           'ambient_light' = 0.2,    # Softer lighting
                                                                           'specular' = 0.2,         # Matte effect
                                                                           'shininess' = 1,          # Low shininess for matte
                                                                           'light_direction' = c(1, 1, 1)))  # Direction of light))

# Export the vis
export(dk_atlas_rois_on_surface, img_only = TRUE, 
       output_img = glue("{github_dir}/plots/DK_atlas_ROIs.png"),
       rglactions = list('shift_hemis_apart'=TRUE,'no_vis'=T), 
       view_angles=fsbrain::get.view.angle.names(angle_set = "t9"))

############ SCHAEFER100 ##############
# 1. Category-selective: "lateraloccipital", "fusiform"
# 2. Visual: "pericalcarine", "lingual"
# 3. Inferior parietal sulcus: "superior"
# 4. Prefrontal cortex: "7Networks_LH_Cont_PFCl_1", "7Networks_RH_Cont_PFCl_1"

# Schaefer100 left/right surface annot
schaef_atlas_dir = "~/data/neuroimaging_atlases/surfaces/Schaefer/fsaverage/"
lh_schaef_annot = freesurferformats::read.fs.annot(glue("{schaef_atlas_dir}/atl-Schaefer2018_space-fsaverage_hemi-L_desc-100Parcels7Networks_deterministic.annot"))
rh_schaef_annot = freesurferformats::read.fs.annot(glue("{schaef_atlas_dir}/atl-Schaefer2018_space-fsaverage_hemi-R_desc-100Parcels7Networks_deterministic.annot"))

lh_schaef_parcels_of_interest <- case_when(lh_schaef_annot$label_names %in% c("7Networks_LH_DorsAttn_Post_1") ~ 1,
                                           lh_schaef_annot$label_names %in% c("7Networks_LH_Vis_4", "7Networks_LH_Vis_5") ~ 2,
                                           lh_schaef_annot$label_names %in% c("7Networks_LH_DorsAttn_Post_3") ~ 3,
                                           lh_schaef_annot$label_names %in% c("7Networks_LH_SalVentAttn_PFCl_1") ~ 4,
                                           T ~ NA_real_)
rh_schaef_parcels_of_interest <- case_when(rh_schaef_annot$label_names %in% c("7Networks_RH_Vis_3") ~ 1,
                                           rh_schaef_annot$label_names %in% c("7Networks_RH_Vis_4", "7Networks_RH_Vis_5") ~ 2,
                                           rh_schaef_annot$label_names %in% c("7Networks_RH_DorsAttn_Post_4") ~ 3,
                                           rh_schaef_annot$label_names %in% c("7Networks_RH_Cont_PFCl_3") ~ 4,
                                           T ~ NA_real_)

schaefer_atlas_rois_on_surface <- vis.symmetric.data.on.subject(subjects_dir, vis_subject_id=subject_id, 
                              morph_data_lh=lh_schaef_parcels_of_interest, 
                              morph_data_rh=rh_schaef_parcels_of_interest, 
                              bg="sulc_light",
                              makecmap_options = list('colFn'=viridis),
                              surface="inflated", draw_colorbar = T,
                              rglactions = list('shift_hemis_apart'=TRUE))
# Export the vis
export(schaefer_atlas_rois_on_surface, img_only = TRUE, 
       output_img = glue("{github_dir}/plots/Schaefer100_atlas_ROIs.png"),
       rglactions = list('shift_hemis_apart'=TRUE,'no_vis'=T), 
       view_angles=fsbrain::get.view.angle.names(angle_set = "t9"))

################## GLASSER ##################
# 1. Category-selective: "LO1", "LO2", "LO3", "FFC"
# 2. Visual: "V1", "V2"
# 3. Inferior parietal sulcus: "IPS1"
# 4. Prefrontal cortex: "46", "9-46d", "a9-46d", "a9-46v"

# Glasser left/right surface annot
glasser_atlas_dir = "~/data/neuroimaging_atlases/surfaces/Glasser_2016/fsaverage/"
lh_glasser_annot = freesurferformats::read.fs.annot(glue("{glasser_atlas_dir}/lh.HCP-MMP1.annot"))
rh_glasser_annot = freesurferformats::read.fs.annot(glue("{glasser_atlas_dir}/rh.HCP-MMP1.annot"))

lh_glasser_parcels_of_interest <- case_when(lh_glasser_annot$label_names %in% c("L_LO1_ROI", "L_LO2_ROI", "L_LO3_ROI", "L_FFC_ROI") ~ 1,
                                            lh_glasser_annot$label_names %in% c("L_V1_ROI", "L_V2_ROI") ~ 2,
                                            lh_glasser_annot$label_names %in% c("L_IPS1_ROI") ~ 3,
                                            lh_glasser_annot$label_names %in% c("L_46_ROI", "L_9-46d_ROI", "L_a9-46d_ROI", "L_a9-46v_ROI") ~ 4,
                                            T ~ NA_real_)
rh_glasser_parcels_of_interest <- case_when(rh_glasser_annot$label_names %in% c("R_LO1_ROI", "R_LO2_ROI", "R_LO3_ROI", "R_FFC_ROI") ~ 1,
                                            rh_glasser_annot$label_names %in% c("R_V1_ROI", "R_V2_ROI") ~ 2,
                                            rh_glasser_annot$label_names %in% c("R_IPS1_ROI") ~ 3,
                                            rh_glasser_annot$label_names %in% c("R_46_ROI", "R_9-46d_ROI", "R_a9-46d_ROI", "R_a9-46v_ROI") ~ 4,
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


ggseg3d(atlas = dk_3d, surface = "inflated", hovertemplate = " ", hemisphere="left") %>% 
  pan_camera("left lateral") %>%
  remove_axes()