**Slide 1**

Good morning everyone. Our project is a SimCLR ablation study, where we investigate which components are important for learning useful visual representations.

**Slide 2**

Before looking at individual results, we use the same evaluation contract: pretrain the encoder with contrastive learning, freeze it, and then test representation quality using linear evaluation.

**Slide 3 4 5**

The first part focuses on data augmentation. This section will explain how different view-generation strengths affect the quality of the learned representation.

**Slide 6 7 8 9**

The next part studies low-label evaluation, showing whether SimCLR pretraining is especially useful when only limited labels are available.

## Slide 10

Now I will talk about two design choices: the projection head and batch size.

For the projection head, SimCLR uses two spaces. The encoder outputs the feature vector **h**, and the projection head maps it to **z**. The contrastive loss is applied to **z**.

The intuition is that **h** should stay useful for downstream classification, while **z** is trained for contrastive separation. This separation reduces the pressure on the encoder representation and can help transfer learning.

## Slide 1

Our results support this idea. With the projection head, the best linear-evaluation accuracy is about **63.6 percent**. Without it, it drops to about **62.2 percent**.

The final accuracy shows the same pattern. The gain is small, but consistent. This suggests that using a separate projection space helps the frozen encoder features transfer better.

## Slide 12

For batch size, we expected larger batches to help because they provide more negative samples for each anchor.

However, the result is not monotonic. Batch size **256** performs best, reaching about **63.5 percent** accuracy. Larger batches perform worse.

This suggests that more negatives do not always help in our lightweight CIFAR-10 setting. Larger batches also change optimization, because there are fewer updates per epoch, and they may include more false negatives from the same class.

Overall, batch size **256** gives the best balance in our runs.

**Slide 13**

To conclude, our results show that SimCLR performance depends on both representation design and training details. The projection head improves transfer, while batch size needs to be matched carefully with optimization settings.