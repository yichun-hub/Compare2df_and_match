# Compare2df_and_match
## Procedure:
  1. Covert new_grouped_nodule_id.json to group dataframe.  
  2. Covert AI predict json file (balance.json) to predict dataframe.  
  3. Compare and Match two dataframes.
## Comparison steps
  Compare two dataframes (group df and predict df) to find Group ID in group df.  

  Comparison steps:   
    * First step: If series ID is the same, combine mask1 and mask2 to be a combination list.  
    * Second step: Iterating mask1 and mask2 in combination list to calculate the overlay area.   
    * Third step:  If the overlay area of two masks are more than 0.0, check the series ID and mask 
                in group df is equal to the caculated mask and return the "Annotation Label" in group df.  
    * Forth step: Also, check the series ID and mask in predict df is equal to the other caculated mask or not.  
                If so, the group id will be add in the "Group ID" column in predict df.  
