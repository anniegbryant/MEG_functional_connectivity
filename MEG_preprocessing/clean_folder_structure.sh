input_model_file=/project/hctsa/annie/github/MEG_functional_connectivity/subject_list_Cogitate_MEG_with_all_data.txt

cat $input_model_file | while read line 
do
   subject=$line
   cd /rds/PRJ-MEG/Cogitate_MEG/sub-${subject}/

   mkdir -p "ses-1"
   mv meg/ ses-1/
   mv anat/ ses-1/

done