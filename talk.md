**Slide 1**

Good morning everyone. Our project is a SimCLR ablation study, where we investigate which components are important for learning useful visual representations.

**Slide 2**

Before looking at individual results, we use the same evaluation contract: pretrain the encoder with contrastive learning, freeze it, and then test representation quality using linear evaluation.

**Slide 3 4 5**

The first part focuses on data augmentation. This section will explain how different view-generation strengths affect the quality of the learned representation.

**Slide 6 7 8 9**

The next part studies low-label evaluation, showing whether SimCLR pretraining is especially useful when only limited labels are available.

## Slide 10

Now I will discuss two design choices: the projection head and batch size.

SimCLR separates the representation into two spaces. The encoder produces h, which is used for downstream tasks. The projection head maps h to z, and the contrastive loss is applied to z.

The intuition is that z can focus on contrastive learning, while h remains more useful for classification and transfer learning.

## Slide 11

Our results support this idea. With the projection head, the best linear-evaluation accuracy is about 63.6 percent. Without it, it drops to about 62.2 percent.

The final accuracy shows the same pattern. The gain is small, but consistent. This suggests that using a separate projection space helps the frozen encoder features transfer better.

## Slide 12

For batch size, we expected larger batches to help because they provide more negative samples.

However, the results are not monotonic. Batch size 256 performs best, with about 63.5 percent accuracy. Larger batches perform worse.

This may be because larger batches reduce the number of updates per epoch and may introduce more false negatives from the same class.

Overall, batch size 256 gives the best balance in our CIFAR-10 experiments.

**Slide 13**

To conclude, our results show that SimCLR performance depends on both representation design and training details. The projection head improves transfer, while batch size needs to be matched carefully with optimization settings.