# Intensity-Based-Template-Matching

Make sure you edit the global constants at the top of main before running.

It will first generate templates which will be written to a templates folder for faster testing. 

Then it calls `test_template_matching()` which runs through (it assumes 20) images running template matching on all of them. It will also compare with the provided ground-truth annotations to check results and provide metrics. 

Answers can be shown as you go by uncommenting `plt.show()` in `test_template_matching()`. They will also be written to a results folder.