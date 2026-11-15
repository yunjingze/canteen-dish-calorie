from pathlib import Path

TEX = """@documentclass[conference]{IEEEtran}
@IEEEoverridecommandlockouts
@usepackage{cite}
@usepackage{amsmath,amssymb,amsfonts}
@usepackage{graphicx}
@usepackage{textcomp}
@usepackage{xcolor}
@usepackage{booktabs}
@usepackage{subcaption}
@usepackage{url}
@graphicspath{{../results/figures/}}
@input{numbers.tex}

@begin{document}

@title{Campus Canteen Dish Recognition with Calorie-Range Estimation: A Controlled Study of Transfer-Learning Depth and Color Augmentation}

@author{@IEEEauthorblockN{Student Name}
@IEEEauthorblockA{School of Computer Science and Technology @\\
University, City, China @\\
student@@example.edu}
}

@maketitle

@begin{abstract}
Photographing a canteen tray and receiving an approximate energy range is a
practical, low-cost way to raise nutritional awareness among students, but it
combines two hard problems: fine-grained food recognition under strong domain
shift, and energy estimation from a single image without volume measurement.
This paper reports a controlled transfer-learning study on a 20-class subset of
Food-101 that is relevant to Chinese campus canteens. We keep the official
Food-101 split, carve a validation set and a held-out probe set out of the
official training images, and build a 1000-image phone-photograph proxy domain
by applying a physically motivated degradation chain (rotation, perspective,
crop, white balance, exposure, blur, vignetting, sensor noise and JPEG
compression) to images that never participate in training or model selection.
Using a single training recipe, we compare a frozen backbone with last-stage
fine-tuning, cross this factor with the presence or absence of photometric
augmentation, and benchmark ResNet-18 against MobileNetV3-Small.
The best ResNet-18 configuration reaches @accTestFour@% top-1 accuracy and
@foneTestFour@% macro-F1 on the 5000-image in-domain test set, but only
@accCanFour@% on the proxy domain, a drop of @gapFour{} points that no amount of
head-only training removes. Adding 69@,500 web images of the same twenty classes,
filtered for label noise and near-duplicates, raises proxy accuracy to
@accCanSeven@% (@gainCanExternal{} points) for @gainExternal{} points in-domain. Unfreezing the last stage improves in-domain
accuracy by @gainFinetuneAug{} points over the frozen backbone trained with the
same augmentation, at no inference cost, whereas color augmentation changes
accuracy by @gainAugFrozen
points when the backbone is frozen and by @gainAugFinetune{} points after
unfreezing. MobileNetV3-Small retains @accTestFive@% accuracy with @paramsFive@,M
parameters and @latGpuFive@,ms per image. Training the same ResNet-18 from
random initialisation instead of ImageNet weights costs @scratchDrop{} points of
in-domain accuracy, so pre-training, not the adaptation policy, is what makes the
task tractable at this data scale. We finally describe a web demo that
returns the top-3 dishes with energy intervals and states the uncertainty of
each entry.
@end{abstract}

@begin{IEEEkeywords}
food recognition, transfer learning, fine-tuning, domain shift, calorie
estimation, Food-101, Grad-CAM
@end{IEEEkeywords}

@section{Introduction}
@IEEEPARstart{C}{anteen} meals are the main source of daily energy for most
Chinese university students, yet the energy content of a dish is rarely
visible at the point of choice. A system that takes a single photograph of a
tray and returns the three most likely dishes together with a rough energy
interval would make calorie awareness immediate, requires no wearable sensor
and no manual logging, and can run on a commodity laptop or an edge device.

Building such a system is not simply an image-classification exercise. Three
difficulties dominate. First, canteen photographs are taken with hand-held
phones under mixed artificial light, at oblique angles, with cluttered trays,
partially eaten portions and steam; a classifier trained on the carefully
curated Food-101 photographs degrades sharply under this shift
@cite{hendrycks2019robustness}. Second, the number of dish categories in a
canteen is large but the frequency distribution is extremely skewed, so
macro-averaged metrics matter more than plain accuracy. Third, energy cannot be
read off an image: portion size is unknown, recipes differ between canteens,
and the mapping from pixels to kilocalories is ill-posed. A system should
therefore output an interval rather than a point estimate, and it should expose
its uncertainty.

This paper takes the position that, for a course-scale project with limited
compute, the most informative questions are not about novel architectures but
about how much of a pretrained network should be adapted and how the training
distribution should be perturbed. We therefore fix one training recipe and vary
only two factors, the fine-tuning depth and the presence of photometric
augmentation, and we measure every model twice: once on the official Food-101
test images and once on a proxy domain that mimics hand-held canteen
photography.

Our contributions are:
@begin{itemize}
@item a reproducible 20-class Food-101 protocol for campus canteens with an
explicit, leak-free split (official training images for training, validation
and the proxy generator; official test images kept untouched for final
reporting) and per-dish energy priors;
@item a controlled comparison of frozen and last-stage fine-tuning crossed
with color augmentation, reported with accuracy, macro-F1, expected
calibration error, cross-domain gap, parameter count, MACs and measured
latency;
@item an external-data experiment that separates photographic diversity from
adaptation policy: 69@,512 label-filtered, near-duplicate-free web images
@cite{lee2018cleannet} raise proxy accuracy by @gainCanExternal{} points, with an
ablation that prices the @leakTest-point inflation near-duplicates would add;
@item a negative result on knowledge distillation from the external-data
teacher into a 1.5@,M-parameter student: the student configurations that differ
only in the loss are statistically indistinguishable, so the cross-domain gain
is attributable to data rather than to the training procedure;
@item a quantitative analysis of the in-domain to proxy-domain gap, including
per-class confusion structure, calibration drift and Grad-CAM evidence;
@item an end-to-end web application that maps a photograph to a top-3 list
with energy intervals, used to produce the demonstration material.
@end{itemize}

@section{Related Work}
@subsection{Food Recognition Benchmarks}
Food-101 @cite{bossard2014food101} contains 101 dish classes with 750 training
and 250 test images per class, collected from food-sharing websites. It remains
the standard pre-training and benchmarking resource for food recognition, and
its official split is the one we preserve. DeepFood @cite{liu2016deepfood}
demonstrated that convolutional networks pretrained on ImageNet and fine-tuned
on food images outperform hand-crafted pipelines for dietary assessment, and
@cite{zhou2016food} reached the same conclusion with a smaller network.
Surveys of food computing @cite{zhang2023foodsurvey} emphasize that
cross-domain generalization, portion estimation and multi-dish scenes remain
open problems, which is precisely the setting of a canteen tray.

@subsection{Transfer Learning and Fine-Tuning Depth}
Features learned on ImageNet @cite{deng2009imagenet} transfer well to
fine-grained targets, but the benefit depends on the similarity between source
and target and on how many layers are adapted @cite{yosinski2014transfer}.
@cite{kornblith2019better} showed that better source models do not always
transfer better, and that the optimal adaptation depth is dataset dependent.
For food images, which share low-level texture and color statistics with
ImageNet but differ in composition, the practical question is whether adapting
only the classifier is enough. We answer it with a controlled ablation rather
than with a new architecture.

@subsection{Calibration, Robustness and Explainability}
Modern networks are over-confident @cite{guo2017calibration}, which is
problematic when a system reports an energy range derived from its confidence.
We therefore report expected calibration error alongside accuracy. Robustness
to common corruptions @cite{hendrycks2019robustness} motivates our proxy
domain, which applies a corruption chain rather than a single synthetic
perturbation. Finally, Grad-CAM @cite{selvaraju2017gradcam} provides
gradient-weighted localization maps that show whether the network attends to the
food or to the tray; we use them offline in Section V to inspect the failures of
the best configuration.

@section{Dataset and Splits}
@subsection{Class Selection}
Food-101 contains 101 classes; a canteen deployed in a Chinese university
serves a much smaller menu. We select 20 classes that are either commonly
served on campus (fried rice, dumplings, ramen, sushi, pancakes, waffles) or
representative of Western fast food that appears in campus cafeterias (pizza,
hamburger, french fries, hot dog, steak, chicken wings, caesar salad, greek
salad, grilled salmon, spaghetti bolognese, lasagna, donuts, ice cream,
chocolate cake). The selection deliberately mixes visually similar pairs
(dumplings vs.@ sushi, pancakes vs.@ waffles, caesar vs.@ greek salad) so that
the confusion structure is informative.

@subsection{Splits}
We keep the official Food-101 partition: the 750 official training images per
class are used for training, validation and proxy generation, and the 250
official test images per class form the in-domain test set (5000 images). From
the official training images of each class we draw, with a fixed seed, 600
training, 100 validation and 50 probe images. The probe images are used only to
generate the proxy domain; they are never seen during training or model
selection, and the validation set alone drives checkpoint selection. Table
@ref{tab:data} summarizes the splits. Because the test images come from the
official test partition, no image used to fit any model appears in the reported
evaluation.

@begin{table}[t]
@centering
@caption{Dataset partitions. All images are disjoint; the proxy domain is
derived from the probe images only.}
@label{tab:data}
@begin{tabular}{llrr}
@toprule
Split & Purpose & Classes & Images @\\
@midrule
Train & parameter fitting & 20 & 12000 @\\
Validation & checkpoint selection & 20 & 2000 @\\
Probe & proxy generation only & 20 & 1000 @\\
Test (official) & in-domain reporting & 20 & 5000 @\\
Canteen proxy & cross-domain reporting & 20 & 1000 @\\
@bottomrule
@end{tabular}
@end{table}

@subsection{Proxy Canteen Domain}
Real canteen photographs cannot be published with this report, so we synthesize
a proxy domain from the probe images. Each image is resized so that its short
side is 512 pixels, then passed through a randomized chain that models a
hand-held phone capture: in-plane rotation of up to $@pm 7^@circ$, a random crop
that keeps 72--92@% of each side, a small perspective warp (up to 3.5@% of the
short side), a warm, fluorescent or cool white-balance cast, brightness in
$[0.72,1.25]$, contrast in $[0.78,1.20]$ and saturation in $[0.70,1.30]$,
Gaussian blur with probability 0.55 (radius 0.4--1.4), motion blur with
probability 0.35, radial vignetting, additive Gaussian noise
($@sigma = 1.5$--$6$), down-sampling to $320@times320$ and JPEG re-compression
with quality 35--65. Figure @ref{fig:pipeline} illustrates the effect. This proxy is not a substitute for a real canteen test
set; it isolates the effect of photometric and geometric degradation while
holding semantics fixed, and Section @ref{sec:lim} explains where real
photographs enter the same evaluation script.

@begin{figure*}[t]
@centering
@includegraphics[width=0.92@textwidth]{architecture.png}
@caption{Pipeline of the proposed system. A photograph is resized to
$224@times224$, passed through a pretrained backbone with a 20-way head, and
the resulting distribution is converted into a top-3 list; each dish carries an
energy prior, and the interval width grows when the posterior is flat.}
@label{fig:pipeline}
@end{figure*}

@subsection{Energy Priors}
Each class is assigned an energy density (kcal per 100@,g) and a typical
canteen portion mass, both taken from public food-composition tables and
rounded to a single representative value. The point estimate for a dish is the
product of the two; the interval is described in Section @ref{sec:calorie}.
These priors are population averages, not measurements of the photographed
plate, and the demo states this limitation explicitly.

@section{Method}
@subsection{Backbones and Adaptation Policies}
We study ResNet-18 @cite{he2016resnet} and MobileNetV3-Small
@cite{howard2019mobilenetv3}, both initialized with ImageNet-1k weights. The
final classifier is replaced by a linear layer with 20 outputs. Two
adaptation policies are compared:
@begin{itemize}
@item @textbf{Frozen}: only the new classifier is trained; all backbone
parameters, including batch-normalization statistics, are fixed. This policy
trains only $10@,260$ parameters, 0.09@% of the $11.19$@,M of ResNet-18.
@item @textbf{Last stage}: the fourth residual stage (or the last inverted-
residual block group for MobileNetV3) and the classifier are trained, the rest
is frozen.
@end{itemize}
@emph{Last stage} is the main configuration because it follows the assignment
requirement of training a new head and then unfreezing the last stage, while
keeping optimization stable on 12@,000 images and leaving inference cost
unchanged. Full-network fine-tuning is supported by the released code but is
not part of the comparison, since it changes neither the parameter count nor the
latency and adds a third axis to an already controlled design.

@subsection{Training Recipe}
All runs share one recipe so that only the intended factors vary: AdamW with
weight decay $5@times10^{-4}$, head learning rate $10^{-3}$, backbone learning
rate $10^{-4}$, cosine annealing over @epochsFour{} epochs, batch size 64, label
smoothing 0.1 @cite{szegedy2016inception}, mixed precision, and input
resolution $224@times224$. The training transform is a random resized crop to
$224@times224$ with scale range $[0.55,1.0]$ followed by a horizontal flip. The
photometric variant adds a color jitter with brightness, contrast and
saturation factors in $[0.65,1.35]$ and hue in $[-0.06,0.06]$. Validation and
test transforms resize the short side to 256, center crop to 224, and normalize
with ImageNet statistics. Checkpoints are selected by validation macro-F1, and
the selected checkpoint is evaluated once on the test and proxy sets.

@subsection{Energy Interval}
@label{sec:calorie}
Let $k_c$ be the energy density of class $c$ in kcal per 100@,g and $m_c$ its
typical portion mass in grams. The point estimate is
@begin{equation}
@hat{E}_c = @frac{k_c@, m_c}{100}.
@end{equation}
Because the portion is unknown, we report
@begin{equation}
@hat{E}_c@left[1-s_c,@; 1+s_c@right], @quad
s_c = 0.15+0.10@cdot@mathbf{1}[p_{@max}<0.55].
@label{eq:interval}
@end{equation}
where $p_{@max}$ is the posterior of the top-1 class. The first term
acknowledges recipe and portion variability, the second widens the interval
when the classifier itself is uncertain, so that a wrong but confident guess is
not presented as a precise number. The same rule is applied to classes ranked
second and third.

@subsection{Metrics}
We report top-1 accuracy and macro-F1 on the in-domain test set and on the
proxy domain, the cross-domain gap (in-domain accuracy minus proxy accuracy),
expected calibration error with 15 equal-width confidence bins, parameter
count, multiply-accumulate operations at $224@times224$ measured with THOP, and
single-image latency on the GPU and on the CPU after warm-up.

@section{Experiments}
@subsection{Setup}
Experiments run on a laptop with an NVIDIA RTX 4060 Laptop GPU (8@,GB), PyTorch
2.11 and CUDA 12.8. Each configuration is trained for @epochsFour{} epochs with a
fixed seed; a run takes @trainTimeFour{} minutes on average (@epochSecFour{}
seconds per epoch). Six configurations are compared: frozen without color
augmentation (E1), frozen with color augmentation (E2), last stage without
(E3), last stage with (E4), and MobileNetV3-Small last stage with color
augmentation (E5). E6 repeats the ResNet-18 recipe with random initialisation
instead of ImageNet weights; because a randomly initialised frozen trunk cannot
learn, E6 trains every parameter (full mode) for the same @epochsSix{} epochs,
which makes it a deliberately generous estimate of what data alone can do.
Two further rows reuse the E4 recipe and change only the data: E7 adds 69@,512
web images of the same twenty classes, and E8 repeats that run without the
near-duplicate filter, which isolates the leakage the filter removes
(Sec.~@ref{sec:extdata}).

@subsection{Main Results}
Table @ref{tab:main} reports the selected checkpoint of every configuration.
The frozen baseline reaches @accTestOne@% accuracy on the official test set,
which already shows how much ImageNet features carry: with only 10@,260
trainable parameters, the classifier separates 20 dish classes far better than
chance. Unfreezing the last stage raises accuracy to @accTestFour@% and macro-F1
to @foneTestFour@%, the best result among the six controlled configurations. On the proxy domain all models
lose a large amount of accuracy, between @gapTwo{} and @gapThree{} points, confirming
that photometric and geometric degradation is the dominant difficulty rather
than class confusion within the clean domain.

@begin{table*}[t]
@centering
@caption{Main results. Accuracy and macro-F1 are percentages on the 5000-image
official test set and on the 1000-image proxy domain; the gap is in percentage
points (test accuracy minus proxy accuracy); ECE is computed on the test set;
time is the wall-clock training time for @epochsFour{} epochs.}
@label{tab:main}
@begin{tabular}{lccccccrr}
@toprule
ID & Configuration & Test acc. & Test F1 & Proxy acc. & Proxy F1 & Gap & ECE & Time (min) @\\
@midrule
E1 & ResNet-18, frozen & @accTestOne & @foneTestOne & @accCanOne & @foneCanOne & @gapOne & @eceTestOne & @trainTimeOne @\\
E2 & ResNet-18, frozen + color & @accTestTwo & @foneTestTwo & @accCanTwo & @foneCanTwo & @gapTwo & @eceTestTwo & @trainTimeTwo @\\
E3 & ResNet-18, last stage & @accTestThree & @foneTestThree & @accCanThree & @foneCanThree & @gapThree & @eceTestThree & @trainTimeThree @\\
E4 & ResNet-18, last stage + color & @accTestFour & @foneTestFour & @accCanFour & @foneCanFour & @gapFour & @eceTestFour & @trainTimeFour @\\
E5 & MobileNetV3-S, last stage + color & @accTestFive & @foneTestFive & @accCanFive & @foneCanFive & @gapFive & @eceTestFive & @trainTimeFive @\\
E6 & ResNet-18, from scratch (full) & @accTestSix & @foneTestSix & @accCanSix & @foneCanSix & @gapSix & @eceTestSix & @trainTimeSix @\\
E7 & E4 + Food-101N ($69@,512$) & @accTestSeven & @foneTestSeven & @accCanSeven & @foneCanSeven & @gapSeven & @eceTestSeven & @trainTimeSeven @\\
E8 & E7 without de-duplication & @accTestEight & @foneTestEight & @accCanEight & @foneCanEight & @gapEight & @eceTestEight & @trainTimeEight @\\
@bottomrule
@end{tabular}
@end{table*}

@subsection{Effect of Fine-Tuning Depth}
Moving from the frozen backbone to the last stage changes in-domain accuracy by
@gainFinetuneNoaug{} points without augmentation (E1 to E3) and by
@gainFinetuneAug{} points with it (E2 to E4), and it lifts macro-F1 in the same
proportion. The improvement is obtained at zero inference cost: the architecture
is identical, and only the number of parameters that receive gradients changes.
Figure @ref{fig:curves} shows that the frozen models converge to a plateau of
about 79@% validation accuracy within four epochs, whereas the last-stage models
keep climbing until the eighth to eleventh epoch, when checkpoint selection
fires; the cosine schedule therefore ends before the last-stage models visibly
overfit, and a longer budget would likely favour them further. The picture is
different across domains: the same intervention moves proxy accuracy only by
@gainCanFinetuneNoaug{} and @gainCanFinetuneAug{} points, so the gap widens from
@gapTwo{} points for the frozen model with colour augmentation to @gapFour{} points
for the fine-tuned one. Most of the in-domain gain is therefore specific to the
clean distribution rather than a general improvement of the features.

@begin{figure*}[t]
@centering
@includegraphics[width=0.95@textwidth]{training_curves.png}
@caption{Training loss (a) and validation accuracy (b) for the six
configurations. Every point is a full pass over the 12@,000 training images.}
@label{fig:curves}
@end{figure*}

@subsection{Effect of Color Augmentation}
Color jitter is intended to make the model invariant to white balance and
exposure, which are precisely the factors that change between the clean and the
proxy domain. In-domain its effect stays within run-to-run noise:
@gainAugFrozen{} points with a frozen backbone and @gainAugFinetune{} points
after unfreezing. On the proxy domain the sign is consistently positive
(@gainCanAugFrozen{} and @gainCanAugFinetune{} points) and augmentation is the
only factor in this study that improves cross-domain accuracy, but the absolute
gain is small next to the @gapFour-point gap it would have to close. The flat
proxy column between E1 and E3, where a @gainFinetuneNoaug-point in-domain gain
translates into @gainCanFinetuneNoaug{} points across domains, suggests that the
bottleneck of the proxy domain is geometric (crop, perspective, resolution) more
than photometric, and a jitter that only perturbs colour cannot address it.

@subsection{Effect of ImageNet Pre-Training}
E6 answers the obvious objection to the other five rows, namely that frozen
features already encode ImageNet, so the comparison says little about the data.
Trained from random initialisation on the same 12@,000 images with every
parameter trainable, the same ResNet-18 reaches @accTestSix@% test accuracy and
@foneTestSix@% macro-F1, i.e.@ @scratchDrop{} points below E4 on the clean
domain, and @accCanSix@% on the proxy, @gapSix-point behind its in-domain score
and @scratchDropCan-point behind E4. Figure @ref{fig:curves} shows the mechanism:
the from-scratch curve is still rising steeply at the end of the budget, having
passed only 67@% training accuracy, whereas the pre-trained models converge
within four to six epochs. Twelve thousand images are therefore not enough to
learn twenty fine-grained categories from pixels, and the value of the frozen
baseline (@accTestOne@%) is the prior that ImageNet supplies, not the 10@,260
trainable parameters. Because E6 updates eleven million parameters and E4 only
the last stage, the $@scratchDrop{}$-point margin is a lower bound on the
pre-training effect rather than an artefact of capacity.

@subsection{Effect of External Data}
@label{sec:extdata}
Every row so far trains on the same 12@,000 images, which is what makes the six
configurations comparable and also what caps any of them. Food-101N
@cite{lee2018cleannet} re-releases the same 101 categories from the web with noisy
labels; restricted to the twenty
classes of this study it contributes 72@,467 candidates. A candidate is kept only
if its released verification label is not $0$ and if its 64-bit dHash differs by
more than a Hamming distance of six from every image of the validation, test and
proxy sets. The filter removes 2@,495 human-verified mislabels and 460
near-duplicates, leaving 69@,512 images, which enter training through a weighted
sampler that draws the official images at weight one and the external ones at
weight one half: an epoch still sees 12@,000 official samples and 69@,512 weighted
external draws, 81@,512 samples in total.

E7 keeps the E4 recipe and changes only the data. In-domain accuracy moves from
@accTestFour@% to @accTestSeven@% (@gainExternal{} points), of the same order as
the color augmentation. On the proxy domain it moves from @accCanFour@% to
@accCanSeven@% (@gainCanExternal{} points), roughly five times the in-domain gain,
and the cross-domain gap contracts from @gapFour{} to @gapSeven{} points. The
largest cross-domain improvement of the study therefore comes not from the
adaptation policy but from the photographic diversity of the data: web images
vary in exposure, white balance, framing and compression far more than the
curated official images, so the model cannot lock onto a single capture style.
That is the effect color jitter approximates on a much smaller scale. The price is
training time: @trainTimeSeven{} minutes for E7 against @trainTimeFour{} for E4,
because an epoch is almost seven times longer. E7 still peaks on the last epoch,
which suggested that the optimization budget rather than the data might be the
limit. It is not: doubling the schedule to 24 epochs (E12, same recipe and data,
cosine annealing over 24 epochs) raises training accuracy from @trainAccSeven@% to
@trainAccTwelve@% but leaves validation at @accValTwelve@% (@accValSeven@% for
E7), test at @accTestTwelve@% (@accTestSeven@%) and the proxy at @accCanTwelve@%
(@accCanSeven@%). The extra epochs are absorbed by the training set, so the recipe
reaches its local optimum well before twelve epochs and the gain of E7 is not an
artefact of an early stop.

E8 asks how much of the in-domain gain is reconstruction leakage. Without the
near-duplicate filter test accuracy rises to @accTestEight@% (@leakTest{} points),
the equivalent of about six of the 5@,000 test images, while proxy accuracy falls
from @accCanSeven@% to @accCanEight@%. The filter therefore costs nothing and E7
is the configuration we report. Calibration moves in the same direction only after
rescaling: raw proxy error is higher for E7 (@eceCanSeven) than for E4
(@eceCanFour), because the validation temperature @tempSeven{} is fitted to a
sharper in-domain distribution, but once scaled the proxy error drops to
@eceCanCalSeven{} against @eceCanCalFour{} for E4.

@subsection{Architecture and Efficiency}
Table @ref{tab:complex} reports complexity and latency. ResNet-18 needs
@paramsFour@,M parameters and @macsFour@,GMACs, whereas MobileNetV3-Small needs
@paramsFive@,M and @macsFive@,GMACs, a reduction of roughly $30@times$ in
arithmetic and $7@times$ in parameters. Latency does not follow the FLOP ratio:
MobileNetV3-Small takes @latGpuFive@,ms against @latGpuFour@,ms for ResNet-18 on
the GPU, and @latCpuFive@,ms against @latCpuFour@,ms on the CPU, so it is
@speedCpuFive@% faster on the CPU but @slowGpuFive@% slower on the GPU. At batch
size one the cost is dominated by the launch and memory-traffic overhead of many
small depthwise layers rather than by arithmetic, so the $30@times$ saving in
FLOPs buys memory and energy rather than uniform wall-clock time, and the
accuracy is @gainMobile{} points lower.

@begin{table}[t]
@centering
@caption{Complexity at $224@times224$ input. Params are in millions, MACs
in GMACs (THOP); latency is the mean of 50 single-image forward passes after 10
warm-up iterations.}
@label{tab:complex}
@small
@begin{tabular*}{@columnwidth}{@@{}lrrrr@@{}}
@toprule
Model & Params & MACs & GPU & CPU @\\
@midrule
ResNet-18 (E4) & @paramsFour & @macsFour & @latGpuFour & @latCpuFour @\\
MobileNetV3-S (E5) & @paramsFive & @macsFive & @latGpuFive & @latCpuFive @\\
@bottomrule
@end{tabular*}
@end{table}

@subsection{Distillation to a Compact Student}
@label{sec:distill}
The architecture comparison above leaves the compact model @gainMobileCost{} points
behind. Knowledge distillation is the obvious remedy: the smaller network is
trained to match both the labels and the softened output distribution of a larger
one. We test it with E7 as the teacher (@accTestSeven@% in-domain, @accCanSeven@%
on the proxy, @paramsFour@,M parameters) and the MobileNetV3-Small student
(@paramsFive@,M), keeping everything else identical to the data-only control
E9: the same 12@,000
official plus 69@,512 external images, the same schedule, seed and augmentation.
Only the loss changes, $@mathcal{L} = (1-@alpha)@,H(y,p) + @alpha@,T^2
@,KL(p_T@,@|@,p^s_T)$, with ($@alpha$,T) = (0.5,4) for E10 and (0.2,3) for E11. Table
@ref{tab:distill} reports all four configurations.

E9 is the control that isolates the data: the same student trained on the same
images without a teacher reaches @accTestNine@% in-domain and @accCanNine@% on the
proxy, @gainDataMobile{} and @gainCanDataMobile{} over E5. Against that control,
distillation does not help. E10 moves test accuracy by @kdOneTest{} points and
proxy accuracy by @kdOneCan{}; E11, with the soft term weighted at $0.2$ instead of $0.5$,
moves them by @kdTwoTest{} and @kdTwoCan{}. Lowering the weight of the soft targets
shrinks both the in-domain loss and the cross-domain gain, which is the signature
of an effect that is not resolvable at this scale: the binomial standard error is
$0.5$ points on the 5@,000 test images and $1.5$ points on the 1@,000 proxy images,
and every difference above is smaller than its own error bar. The gap contracts
from @gapFive{} to @gapNine{} points with the data alone; distillation adds nothing
measurable to that contraction.

Three properties of the setup explain the outcome. The teacher is only eight points
stronger than the student and was trained on the same images with the same recipe,
so the dark knowledge it can transfer is the weakest form of the signal. The
student is capacity-limited rather than label-limited: it saturates near 78.5@%
validation accuracy with or without a teacher, and its training accuracy moves only
from 66.8@% to 67.2@% between E10 and E9, so the soft targets neither starve nor
unlock it. And the effect that does survive, the @gainCanDataMobile{}-point proxy
gain from the external images, is larger than any distillation effect we could
detect at this student size on curated data. Calibration is unaffected: after
temperature scaling the test error of the three students converges to
@eceTestCalNine{}, @eceTestCalTen{} and @eceTestCalEleven{}. We therefore report
distillation as a negative result and read it as further evidence that at this data
scale the binding constraint is the photographic distribution of the training
images rather than the training procedure.

@begin{table}[t]
@centering
@caption{Distillation from the E7 ResNet-18 teacher into MobileNetV3-Small on the
same 81@,512-sample mixture. E9 is the data-only control; the distilled runs differ
from it only in the loss.}
@label{tab:distill}
@small
@begin{tabular*}{@columnwidth}{@@{}lrrrr@@{}}
@toprule
Configuration & Test & Proxy & Gap & ECE @\\
@midrule
E5 (official only) & @accTestFive & @accCanFive & @gapFive & @eceTestFive @\\
E9 (data only) & @accTestNine & @accCanNine & @gapNine & @eceTestNine @\\
E10 ($@alpha$=0.5, T=4) & @accTestTen & @accCanTen & @gapTen & @eceTestTen @\\
E11 ($@alpha$=0.2, T=3) & @accTestEleven & @accCanEleven & @gapEleven & @eceTestEleven @\\
@bottomrule
@end{tabular*}
@end{table}

@subsection{Cross-Domain Behaviour}
@label{sec:crossdomain}
Figure @ref{fig:confusion} contrasts the confusion matrices of E4 on the test
and proxy sets. In-domain errors concentrate on the visually similar pairs that
motivated the class selection: dumplings and sushi share a pale, low-contrast
appearance, pancakes and waffles share a gridded texture, and the two salads
share a bowl of green leaves. On the proxy domain the diagonal loses mass, the
same semantic pairs survive (caesar to greek salad, steak to grilled salmon),
and new pairs that share no semantics appear (grilled salmon to caesar salad,
hamburger to french fries, ice cream to chocolate cake). Recall collapses
unevenly across classes: french fries keeps 0.84 while hamburger drops to 0.18,
which is consistent with the model losing the evidence for the class-defining
texture and falling back on colour and context, cues that the degradation
preserves.

Figure @ref{fig:calibration} shows the reliability diagrams of all six models.
Both domains stay above the diagonal, so label smoothing combined with twelve
epochs of training leaves the models under-confident rather than over-confident,
and the fine-tuned models are the better calibrated ones in both domains (ECE
@eceTestFour{} and @eceCanFour). On the proxy domain the mean confidence falls
from 0.781 to 0.583 while accuracy falls from @accTestFour@% to @accCanFour@%,
so the two move together; 48@% of the proxy images have a posterior below the
0.55 threshold and therefore receive a widened energy interval, which is exactly
the behaviour the interface was designed to show. The frozen models start from a
larger calibration error (@eceTestOne{} in-domain, @eceCanOne{} on the proxy) and
do not benefit from this coupling.

Because the models are under-confident rather than over-confident, post-hoc
temperature scaling (Guo et al., 2017) corrects them: one scalar fitted on the
validation split, $T=@tempFour{}$ for E4, sharpens the posteriors and lowers ECE
from @eceTestFour{} to @eceTestCalFour on the test set and from @eceCanFour{} to
@eceCanCalFour on the proxy, without changing a single predicted class. The
dashed curves of Figure @ref{fig:calibration} are the recalibrated E4
reliability diagram. The temperature is stored next to the checkpoint and applied
by the demo before the top-3 list is built, so the confidence shown to the user
is the calibrated one. A value below one quantifies the under-confidence noted above.

Two inference-time questions remain. The first is whether test-time augmentation
closes part of the gap. Averaging the softmax over a horizontal flip raises E7
from @accTestSeven@% to @ttaHflipTestSeven@% in-domain and from @accCanSeven@% to
@ttaHflipCanSeven@% on the proxy; a five-crop average reaches @ttaFiveTestSeven@%
and @ttaFiveCanSeven@%, while multi-scale averaging (short side 256, 288 and 320)
reaches @ttaScalesTestSeven@% in-domain but falls to @ttaScalesCanSeven@% on the
proxy. The pattern follows the mechanism above: the proxy is defined by blur and
downsampling, so re-rendering the input at a larger short side removes part of the
evidence the proxy was designed to withhold, whereas a flip leaves the image
statistics unchanged. Test-time augmentation is therefore worth about one point,
not the @gapSeven-point gap, and the multi-scale variant is harmful on this domain.

The second question is how much one draw of the proxy costs. The proxy is a single
random degradation of the 1@,000 clean images, so its level is itself a sample.
Re-running the same generator with a different seed lowers E4 from @accCanFour@%
to @redrawCanFour@% and E7 from @accCanSeven@% to @redrawCanSeven@%, a spread of
two to four points, while the difference between the two models contracts only
from @gainCanExternal{} to @redrawGapSeven{} points. Comparisons between models on
this proxy are therefore meaningful; comparisons of the absolute level with other
studies are not, which is why the gap is reported alongside the level throughout.

Because every number here is a point estimate on a finite sample, we also report
intervals and paired tests rather than bare differences. Bootstrapping the 5,000
test images (4,000 resamples) gives E7 91.7@% (95@% CI 90.9--92.4) against E4
90.5@% (89.7--91.4), and an exact McNemar test on the same images fixes 223 that
E4 misses and breaks 167 that E4 gets, $p=0.005$. On the two proxy draws the same
comparison reads 71.6@% against 66.0@% ($p<0.001$) and 67.6@% against 63.6@%
($p=0.005$), so under the single-view protocol the external-data gain survives
both draws; it is the only intervention in the study that does. De-duplication is
not detectable: E8 against E7 flips 119/125, 65/58 and 63/72 images on test,
proxy and the redraw, with $p$ between 0.49 and 0.75. An ensemble of the five
fine-tuned models is reliably better than any one of them---93.4@% in-domain,
74.2@% and 71.4@% on the two proxy draws, $p@le0.003$ against E7 throughout---and
is left out of the deployment only because it costs five forward passes; it is
the first item of the deployment backlog. On the 100 real photographs E7 scores
91@% (85--96) under this protocol and 95@% (90--99) with the six-view fusion the
demo uses; both intervals overlap the test-set interval, which is the statistical
form of the observation that the batch behaves like a clean photograph rather than
like a canteen tray.

@begin{figure*}[t]
@centering
@begin{subfigure}{0.48@textwidth}
@includegraphics[width=@linewidth]{confusion_test.png}
@caption{In-domain test set}
@end{subfigure}
@hfill
@begin{subfigure}{0.48@textwidth}
@includegraphics[width=@linewidth]{confusion_canteen.png}
@caption{Proxy canteen domain}
@end{subfigure}
@caption{Row-normalized confusion matrices of the best configuration (E4).
Class order follows the class list in the released code.}
@label{fig:confusion}
@end{figure*}

@begin{figure*}[t]
@centering
@includegraphics[width=0.92@textwidth]{calibration.png}
@caption{Reliability diagrams of the six configurations on the test (a) and the
proxy (b) domains. The dashed line is perfect calibration; ECE is computed with
15 equal-width bins.}
@label{fig:calibration}
@end{figure*}

@subsection{Qualitative Analysis}
Figure @ref{fig:qual} shows Grad-CAM maps and the top-1 prediction with its
energy interval for six proxy images. For correctly classified dishes the
evidence concentrates on the food itself (the noodles in the fried rice, the
folded dough of the dumplings, the fish on the sushi rice); for the images whose
map is diffuse, the activation spreads over the plate rim and the background,
which is the visual counterpart of the loss of evidence measured in
Figure @ref{fig:confusion}. The top-3 list absorbs much of the error: on the
proxy the correct class appears in the top-3 for 85@% of the images although
top-1 accuracy is @accCanFour@%, and the interface flags the 48@% of cases whose
posterior falls below the widening threshold.

@begin{figure*}[t]
@centering
@begin{subfigure}{0.38@textwidth}
@includegraphics[width=@linewidth]{gradcam_canteen.png}
@caption{Grad-CAM overlays}
@end{subfigure}
@hfill
@begin{subfigure}{0.38@textwidth}
@includegraphics[width=@linewidth]{predictions_canteen.png}
@caption{Top-1 prediction and energy interval}
@end{subfigure}
@caption{Qualitative results on the proxy domain for the best configuration
(E4).}
@label{fig:qual}
@end{figure*}

@subsection{Deployed Web Interface}
The pipeline is a small web service: a FastAPI backend loads the checkpoint once,
and a single page lets the user upload a photograph or pick a bundled proxy
sample. The response contains the
top-3 classes with their posteriors, the energy interval of (2) for each entry,
and the uncertainty flag. Figure @ref{fig:demo} is a screenshot of the running
system on a proxy image of Caesar salad: the correct dish is ranked first with a
posterior of 0.98 and an interval of 220--300@,kcal, while the runner-up (Greek
salad) stays below 0.1; with the weight field set to 350@,g every entry is
rescaled to the typed mass, giving 390--520@,kcal here. A forward pass takes
0.01@,s on the RTX 4060, within the budget of a phone browser.

The interface also exposes the portion uncertainty that the interval of (2)
hides. A three-way selector (small, regular, large) multiplies the reference mass
by 0.7, 1.0 and 1.35, and a weight field accepts the actual mass in grams; either
control rescales every entry on the client from the per-100@,g prior, without a
new forward pass, so the user sees immediately how much of the reported range is
recognition uncertainty and how much is the serving size.

A second tab handles mixed trays. A multi-label variant of the same backbone
(20 independent sigmoids, trained on plates composited from two or three dishes)
marks which items are present, and each detected part gets its own selector or
weight, so the page can sum the parts into one tray total.

The subset is also enforced as a soft constraint. Because a 20-way softmax always
returns a class, a whole roast chicken comes back as grilled salmon at 49@%.
Scoring three crops (centre, squashed full frame, wide) and keeping the worst
top-1 probability separates the two cases: supported dishes stay confident in
every framing, unseen ones do not. At 0.32 the gate flags 5.3@% of supported
photos and 27@% of 20 unseen classes, and the page then says the photo is probably
not one of the twenty rather than showing a confident wrong answer.

The rule above is a heuristic on scores, and two failure modes exposed it. A
photograph of a supermarket product---a frozen-pizza carton whose front shows a
pizza---is confidently in-domain, because nothing in the pipeline asks whether
the pixels depict a plated dish rather than a printed one: in an audit of 62
product photographs, 36 were accepted and 12 were reported as the dish printed on
the package. And the worst-case-of-three-framings score does not separate food
from non-food: over 131 photographs that contain no dish (empty trays, cups,
napkins, menu boards, hands) its median is 0.32 against 0.94 for real dishes, and
no single threshold both rejects the negatives and keeps the dishes---at 0.32,
70 of the 131 negatives are accepted; at 0.66, only 66 of 86 dishes survive. Both
problems are handled by learned vetoes instead, each a logistic regression on the
512-dimensional average-pooled feature that the classifier already computes for
its full-frame view, so neither head costs an extra forward pass. The food veto
is fitted on the 131 negatives and 3,700 dish images and rejects 96.9@% (95@% CI
93.9--99.2) of the negatives while keeping 95@% (90--99) of 100 real canteen
dishes; the packaging veto is fitted on 79 product photographs and rejects all 62
audited cartons before scoring, at the price of 18 of 3,700 dish images (0.5@%),
mostly ice cream and doughnuts photographed on white. Both are stored as a small
JSON weight file next to the checkpoint, and the page names which veto fired.

@begin{figure}[t]
@centering
@includegraphics[width=0.93@columnwidth]{demo_ui.png}
@caption{The web demo: a photo (left) yields the top-3 dishes with posteriors and
energy intervals (right); low-confidence entries are marked.}
@label{fig:demo}
@end{figure}

@subsection{Discussion and Limitations}
@label{sec:lim}
Three limitations matter. First, the cross-domain numbers come from a
synthetic proxy built from Food-101 images, not from real canteen photographs;
the degradation chain covers the main photometric and geometric effects of a
phone capture but not occlusion by a tray, overlapping servings, or steam. The
external-data experiment is a partial remedy rather than a substitute: 69@,512
web images recover @gainCanExternal{} of the @gapFour-point gap (E7), which says
that photographic diversity is part of what the synthetic proxy was missing, but
they are still curated photos, not unprompted canteen snapshots. The level of the
proxy is also a single random draw, so only the gap between models, which is much
more stable than the level, is interpreted (Section @ref{sec:crossdomain}).
Second, energy is estimated from class-level priors and a reference portion mass:
the interval covers recipe variability, and the selector or a typed weight fixes
the mass the user assumes, but the demo never measures it from the image.
Third, every configuration is trained from a single seed. Repeating E4 with a
different seed moves test accuracy from @accTestFour@% to @accTestFourSeed@% and
proxy accuracy from @accCanFour@% to @accCanFourSeed@%, about half a point, which
is the size of the augmentation, mobile and distillation differences discussed
above; only the cross-domain effect of external data clears that bar. The
bootstrap intervals of Section @ref{sec:crossdomain} make the same point from the
other side: on 1,000 proxy images an interval spans about three points, which is
why the paired tests, and not the point estimates, decide which differences are
real. The 24-epoch
check (E12) shows that the budget is not the limiting factor either: it adds five
points of training accuracy without moving validation, test or proxy. The
fine-tuning depth remains partially confounded with the learning-rate schedule,
which the seed check does not remove.

@section{Conclusion}
We presented a reproducible 20-class protocol for campus canteen dish
recognition with calorie-range estimation, and used it to isolate the effect of
fine-tuning depth and color augmentation under a fixed recipe. Unfreezing the
last residual stage helps most: +@gainFinetuneAug{} points over the frozen
backbone at no inference cost, while color augmentation gains less and
from-scratch training loses @scratchDrop{} points, locating most of the accuracy
in the pre-trained weights. The largest remaining error source is domain shift:
accuracy drops by @gapFour{} points on a phone-photograph proxy despite unchanged
class semantics, and adding 69@,512 web images of the same classes recovers
@gainCanExternal{} of those points, more than any change of the adaptation
policy. A web demo returning top-3 dishes with energy intervals completes the
pipeline; two small learned vetoes sit in front of it, so that photographs of
product packaging and of food-free scenes are named as such instead of being
given a confident wrong label.

@section*{Acknowledgment}
The author thanks the course staff for the assignment specification and the
maintainers of the Food-101 dataset.

@bibliographystyle{IEEEtran}
@bibliography{references}

@end{document}
"""

out = Path(__file__).with_name("main.tex")
out.write_text(
    TEX.replace("@@", "\x00").replace("@", chr(92)).replace("\x00", "@"),
    encoding="utf-8",
)
print("wrote", out, len(TEX))
