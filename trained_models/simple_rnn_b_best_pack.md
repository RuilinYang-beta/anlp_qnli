Observations compared to `simple_rnn_b_best_noPack.md`:

- till epoch ~100, the same
- after epoch ~100, two files diverge, even with random seed
  - there must be something else that introduces randomness
- time consumed still higher compared to noPack
- However, the range of loss is reasonable, so likely the implementation is correct
- Conclusion: don't use pack

Epoch-0, avg loss per example in epoch 1.1084649562835693
Epoch-1, avg loss per example in epoch 1.1015238761901855
Epoch-2, avg loss per example in epoch 1.09715735912323
Epoch-3, avg loss per example in epoch 1.0940741300582886
Epoch-4, avg loss per example in epoch 1.0916980504989624
Epoch-5, avg loss per example in epoch 1.0898610353469849
Epoch-6, avg loss per example in epoch 1.0883331298828125
Epoch-7, avg loss per example in epoch 1.0870733261108398
Epoch-8, avg loss per example in epoch 1.0859475135803223
Epoch-9, avg loss per example in epoch 1.0849831104278564
Epoch-10, avg loss per example in epoch 1.084082841873169
Epoch-11, avg loss per example in epoch 1.0833830833435059
Epoch-12, avg loss per example in epoch 1.0826301574707031
Epoch-13, avg loss per example in epoch 1.0820937156677246
Epoch-14, avg loss per example in epoch 1.0814779996871948
Epoch-15, avg loss per example in epoch 1.0809348821640015
Epoch-16, avg loss per example in epoch 1.080382227897644
Epoch-17, avg loss per example in epoch 1.0799388885498047
Epoch-18, avg loss per example in epoch 1.0793992280960083
Epoch-19, avg loss per example in epoch 1.078908920288086
Epoch-20, avg loss per example in epoch 1.0784168243408203
Epoch-21, avg loss per example in epoch 1.0780212879180908
Epoch-22, avg loss per example in epoch 1.077593207359314
Epoch-23, avg loss per example in epoch 1.07713782787323
Epoch-24, avg loss per example in epoch 1.0767481327056885
Epoch-25, avg loss per example in epoch 1.0761609077453613
Epoch-26, avg loss per example in epoch 1.0756313800811768
Epoch-27, avg loss per example in epoch 1.07515287399292
Epoch-28, avg loss per example in epoch 1.0746723413467407
Epoch-29, avg loss per example in epoch 1.0741288661956787
Epoch-30, avg loss per example in epoch 1.0734517574310303
Epoch-31, avg loss per example in epoch 1.0728766918182373
Epoch-32, avg loss per example in epoch 1.0721995830535889
Epoch-33, avg loss per example in epoch 1.0713905096054077
Epoch-34, avg loss per example in epoch 1.0705987215042114
Epoch-35, avg loss per example in epoch 1.0696080923080444
Epoch-36, avg loss per example in epoch 1.068561315536499
Epoch-37, avg loss per example in epoch 1.0673744678497314
Epoch-38, avg loss per example in epoch 1.066024661064148
Epoch-39, avg loss per example in epoch 1.0644946098327637
Epoch-40, avg loss per example in epoch 1.0627169609069824
Epoch-41, avg loss per example in epoch 1.0604947805404663
Epoch-42, avg loss per example in epoch 1.0579943656921387
Epoch-43, avg loss per example in epoch 1.0554122924804688
Epoch-44, avg loss per example in epoch 1.0527362823486328
Epoch-45, avg loss per example in epoch 1.0501657724380493
Epoch-46, avg loss per example in epoch 1.047788381576538
Epoch-47, avg loss per example in epoch 1.0458226203918457
Epoch-48, avg loss per example in epoch 1.0434647798538208
Epoch-49, avg loss per example in epoch 1.0417065620422363
Epoch-50, avg loss per example in epoch 1.0395622253417969
Epoch-51, avg loss per example in epoch 1.0376591682434082
Epoch-52, avg loss per example in epoch 1.0360071659088135
Epoch-53, avg loss per example in epoch 1.0333582162857056
Epoch-54, avg loss per example in epoch 1.0319874286651611
Epoch-55, avg loss per example in epoch 1.030832052230835
Epoch-56, avg loss per example in epoch 1.0286743640899658
Epoch-57, avg loss per example in epoch 1.0262035131454468
Epoch-58, avg loss per example in epoch 1.0245270729064941
Epoch-59, avg loss per example in epoch 1.022939920425415
Epoch-60, avg loss per example in epoch 1.0200130939483643
Epoch-61, avg loss per example in epoch 1.017601728439331
Epoch-62, avg loss per example in epoch 1.015838861465454
Epoch-63, avg loss per example in epoch 1.0129303932189941
Epoch-64, avg loss per example in epoch 1.0103963613510132
Epoch-65, avg loss per example in epoch 1.0090577602386475
Epoch-66, avg loss per example in epoch 1.0045932531356812
Epoch-67, avg loss per example in epoch 1.0018564462661743
Epoch-68, avg loss per example in epoch 1.0000196695327759
Epoch-69, avg loss per example in epoch 0.9942502975463867
Epoch-70, avg loss per example in epoch 0.9916051626205444
Epoch-71, avg loss per example in epoch 0.9863697290420532
Epoch-72, avg loss per example in epoch 0.980503261089325
Epoch-73, avg loss per example in epoch 0.9770370125770569
Epoch-74, avg loss per example in epoch 0.9733190536499023
Epoch-75, avg loss per example in epoch 0.960204541683197
Epoch-76, avg loss per example in epoch 0.9653639793395996
Epoch-77, avg loss per example in epoch 0.9702291488647461
Epoch-78, avg loss per example in epoch 0.9592469930648804
Epoch-79, avg loss per example in epoch 0.9288748502731323
Epoch-80, avg loss per example in epoch 0.927476704120636
Epoch-81, avg loss per example in epoch 0.9119938611984253
Epoch-82, avg loss per example in epoch 0.9144807457923889
Epoch-83, avg loss per example in epoch 0.9216309189796448
Epoch-84, avg loss per example in epoch 0.9234250783920288
Epoch-85, avg loss per example in epoch 0.8876357078552246
Epoch-86, avg loss per example in epoch 0.8690204620361328
Epoch-87, avg loss per example in epoch 0.8988486528396606
Epoch-88, avg loss per example in epoch 0.870324432849884
Epoch-89, avg loss per example in epoch 0.8662083148956299
Epoch-90, avg loss per example in epoch 0.8592227697372437
Epoch-91, avg loss per example in epoch 0.8495973944664001
Epoch-92, avg loss per example in epoch 0.8619178533554077
Epoch-93, avg loss per example in epoch 0.8755791783332825
Epoch-94, avg loss per example in epoch 0.8505334258079529
Epoch-95, avg loss per example in epoch 0.8664894700050354
Epoch-96, avg loss per example in epoch 0.8450393676757812
Epoch-97, avg loss per example in epoch 0.8377940654754639
Epoch-98, avg loss per example in epoch 0.8361518979072571
Epoch-99, avg loss per example in epoch 0.8366268277168274
Epoch-100, avg loss per example in epoch 0.8300588726997375
Epoch-101, avg loss per example in epoch 0.8330938220024109
Epoch-102, avg loss per example in epoch 0.9552395939826965
Epoch-103, avg loss per example in epoch 1.1161952018737793
Epoch-104, avg loss per example in epoch 1.039521336555481
Epoch-105, avg loss per example in epoch 0.9451130628585815
Epoch-106, avg loss per example in epoch 0.8662058711051941
Epoch-107, avg loss per example in epoch 0.8472031950950623
Epoch-108, avg loss per example in epoch 0.835995078086853
Epoch-109, avg loss per example in epoch 1.0855801105499268
Epoch-110, avg loss per example in epoch 1.0787752866744995
Epoch-111, avg loss per example in epoch 1.0768824815750122
Epoch-112, avg loss per example in epoch 1.0759079456329346
Epoch-113, avg loss per example in epoch 1.074097990989685
Epoch-114, avg loss per example in epoch 1.073306918144226
Epoch-115, avg loss per example in epoch 1.072087049484253
Epoch-116, avg loss per example in epoch 1.0716227293014526
Epoch-117, avg loss per example in epoch 1.0704677104949951
Epoch-118, avg loss per example in epoch 1.0696569681167603
Epoch-119, avg loss per example in epoch 1.0688155889511108
Epoch-120, avg loss per example in epoch 1.0678036212921143
Epoch-121, avg loss per example in epoch 1.0665472745895386
Epoch-122, avg loss per example in epoch 1.0658766031265259
Epoch-123, avg loss per example in epoch 1.0652151107788086
Epoch-124, avg loss per example in epoch 1.0648574829101562
Epoch-125, avg loss per example in epoch 1.0637497901916504
Epoch-126, avg loss per example in epoch 1.0632497072219849
Epoch-127, avg loss per example in epoch 1.06240713596344
Epoch-128, avg loss per example in epoch 1.0613610744476318
Epoch-129, avg loss per example in epoch 1.0603370666503906
Epoch-130, avg loss per example in epoch 1.0588185787200928
Epoch-131, avg loss per example in epoch 1.0576896667480469
Epoch-132, avg loss per example in epoch 1.0567753314971924
Epoch-133, avg loss per example in epoch 1.0553135871887207
Epoch-134, avg loss per example in epoch 1.0543103218078613
Epoch-135, avg loss per example in epoch 1.0528216361999512
Epoch-136, avg loss per example in epoch 1.0516297817230225
Epoch-137, avg loss per example in epoch 1.050009846687317
Epoch-138, avg loss per example in epoch 1.048244833946228
Epoch-139, avg loss per example in epoch 1.0465587377548218
Epoch-140, avg loss per example in epoch 1.0443363189697266
Epoch-141, avg loss per example in epoch 1.0414725542068481
Epoch-142, avg loss per example in epoch 1.039263129234314
Epoch-143, avg loss per example in epoch 1.0352717638015747
Epoch-144, avg loss per example in epoch 1.0315098762512207
------------ batch training with no pack_pad, pad_pack; supposedly slow, but fast ------------
Took : 428.42864537239075 seconds;
Device : cuda;
Epoch : 145;
Learning rate : 0.0001;
Batch size : 300;
Embedding dim : 32;
Hidden size : 50;
------------------ use the freshly trained model ------------------
[train] Loss: 1.0301182736393106, Accuracy: 0.45003861003861007, F1-macro: 0.3993666386968378
[dev] Loss: 1.0726059070260254, Accuracy: 0.41855670103092785, F1-macro: 0.3737129865598153
------------------ load the saved model ------------------
[train] Loss: 1.0301182736393106, Accuracy: 0.45003861003861007, F1-macro: 0.3993666386968378
[dev] Loss: 1.0726059070260254, Accuracy: 0.41855670103092785, F1-macro: 0.3737129865598153
