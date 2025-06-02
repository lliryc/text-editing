# Error Analysis

This analysis evaluates the performance of different models across specific error types on the development sets of **QALB-2014**, **ZAEBUC**, and **MADAR CODA**. The **evaluated models** are:
- **Best Seq2Seq++ baselines**:
  - AraT5+Morph+GED<sup>43</sup> for QALB-2014  
  - AraBART+Morph+GED<sup>13</sup> for ZAEBUC  
  - AraT5+City for MADAR CODA  

- **Best text editing models**:
  - SWEET<sup>2</sup><sub>NoPnx</sub>+SWEET<sub>Pnx</sub> for QALB-2014 and ZAEBUC  
  - SWEET for MADAR CODA  

- **4-Ensemble** on all datasets

We do so by first aligning the erroneous input sentences with the models' outputs and passing the alignments to ARETA to obtain the specific error types.
We then project the error types on the tokens and evaluate those error tags against the gold error tags. Running the [error_analysis.sh](error_analysis.sh) script does all of these steps automatically.
