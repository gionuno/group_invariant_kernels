# group_invariant_kernels
Group Invariant Kernels, based on Learning with Group Invariant Features by Youssef Mroueh, et al.

By applying transformations on noisy samples:

![image](transformations.png)

This one calculates a bunch of histograms of random transformations of random samples to approximate a Harr Integrated Group Invariant Kernel function.

![image](image_and_phi.png)

![image](phi_barplots.png)

It's easier to say that it transforms the random sample in a invertible manner, calculates a PDF of the dot product and the input over a range from [-1,1].

Really neat stuff!
