**Slide 1**

Good morning everyone. Our project is a SimCLR ablation study, where we investigate which components are important for learning useful visual representations.

**Slide 2**

Before looking at individual results, we use the same evaluation contract: pretrain the encoder with contrastive learning, freeze it, and then test representation quality using linear evaluation.

**Slide 3**

The first part focuses on data augmentation. This section will explain how different view-generation strengths affect the quality of the learned representation.

**Slide 4**

The next part studies low-label evaluation, showing whether SimCLR pretraining is especially useful when only limited labels are available.

**Slide 5**

Now I will talk about two design choices: the projection head and batch size.

For the projection head, the key idea is that SimCLR uses two different spaces. The encoder outputs a 512-dimensional feature vector, called h. With the projection head, h is mapped into another vector z, and the NT-Xent contrastive loss is applied to z.

This means h can stay useful for downstream classification, while z is optimized for contrastive separation. Without the projection head, the same feature h has to satisfy both purposes, which may reduce transfer quality.

**Slide 6**

The result supports this interpretation. With the projection head, the best linear-evaluation accuracy is 63.59 percent. Without the projection head, it drops to 62.16 percent, so the improvement is 1.43 percentage points.

The final accuracy shows the same trend: 62.84 percent with the head, compared with 61.40 percent without it. The gain is not huge, but it is consistent, suggesting that separating the contrastive space from the encoder feature space helps the frozen representation transfer better.

**Slide 7**

For batch size, we expected larger batches to help because each anchor gets more negative samples. With two augmented views per image, the number of negatives is two times batch size minus two.

However, our result is not monotonic. Batch size 256 performs best, with 510 negatives per anchor and 63.47 percent best accuracy. Larger batches, 512 and 1024, actually perform worse.

So in our lightweight CIFAR-10 setting, more negatives are not automatically better. Batch size also changes optimization: larger batches have fewer updates per epoch, use the same learning-rate schedule, and may include more false negatives from the same class. Overall, batch size 256 gives the best balance in our runs.

**Slide 8**

To conclude, our results show that SimCLR performance depends on both representation design and training details. The projection head improves transfer, while batch size needs to be matched carefully with optimization settings.