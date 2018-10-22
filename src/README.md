# Intro
Kindly read the Proposal first to get the idea of the use of the code.

---

# Usage
The `src.py` file contains some of the custom function definitions, which are used in `demo.py`.
Run `demo.py` for a sample run. For now, the whole implementation isn't complete and currently only the epipolar lines are being printed, in future, we would add lookup table and highlight the valid TERs.

---

# Dataset
For now we are using a custom dataset in which we artificially move an object along a linear path and take a number of images and try to find the valid TERs (Temporal Epipolar Regions) in this case.

---

# Requirements
```
python 3.7.0
opencv-python 3.4.2.16
opencv-contrib-python 3.4.2.16
numpy 1.15.2
scipy 1.1.0
sklearn 0.19.2
```
---

# Idea
This implementation tries to implement the algorithm to find the valid TERs in an image where we need to find the search space of correspondance of a point of interest, given initial correspondences in minimum of 3 images that are different from this image.

---

# Methodology
See the proposal for the methodology.

---

# Testing
Yet to be done

---

# Results
Yet to be done

---

# Conclusions
Yet to be done

---