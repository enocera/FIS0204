(* ::Package:: *)

(* ::Subsection::Closed:: *)
(*Perceptron*)


sig[z_]:=HeavisideTheta[z]
a2[x1_,x2_,t1_,t2_,b_]:=sig[x1*t1+x2*t2+b]


Manipulate[
  Plot3D[a2[x1,x2,t1,t2,b1],{x1,-2,2},{x2,-2,2}],
{t1,-1,1},{t2,-1,1},{b1,-1,1}
]


(* ::Subsection::Closed:: *)
(*Hidden layer ANN*)


sig[z_]:=Tanh[z]


a1[x1_,t_,b_]:=sig[x*t+b]
a2[x1_,x2_,t1_,t2_,b_]:=sig[x1*t1+x2*t2+b]


a2[a1[x,t1,b1],a1[x,t2,b2],t3,t4,b3]


Manipulate[
  Plot[a2[a1[x,t1,b1],a1[x,t2,b2],t3,t4,b3],{x,-10,10}],
{t1,-1,1},{t2,-1,1},{t3,-1,1},{t4,-1,1},
{b1,-1,1},{b2,-1,1},{b3,-1,1}
]


a4[x1_,x2_,x3_,x4_,t1_,t2_,t3_,t4_,b_]:=sig[x1*t1+x2*t2+x3*t3+x4*t4+b]


Manipulate[
  Plot[a4[a1[x,t1,b1],a1[x,t2,b2],a1[x,t3,b3],a1[x,t4,b4],t5,t6,t7,t8,b5],{x,-10,10}],
{t1,-1,1},{t2,-1,1},{t3,-1,1},{t4,-1,1},{t5,-1,1},{t6,-1,1},{t7,-1,1},{t8,-1,1},
{b1,-1,1},{b2,-1,1},{b3,-1,1},{b4,-1,1},{b5,-1,1}
]


(*sig[z_]:=HeavisideTheta[z]*)
sig[z_]:=1/(1+Exp[-z])
a2[x1_,x2_,t1_,t2_,b_]:=sig[x1*t1+x2*t2+b]
a2[a2[x1,x2,t1,t2,b1],a2[x1,x2,t3,t4,b2],t5,t6,b3]


Manipulate[
  Plot3D[a2[a2[x1,x2,t1,t2,b1],a2[x1,x2,t3,t4,b2],t5,t6,b3],{x1,-20,20},{x2,-20,20}],
{t1,-1,1},{t2,-1,1},{t3,-1,1},{t4,-1,1},{t5,-1,1},{t6,-1,1},
{b1,-1,1},{b2,-1,1},{b3,-1,1}
]


(* ::Subsection::Closed:: *)
(*A simple example of Backpropagation*)


(* ANN: 2 input nodes, 1 hidden layer of 3 nodes, 1 output node *)


weights = {ww[2,1,1],ww[2,1,2],ww[2,2,1],ww[2,2,2],ww[2,3,1],ww[2,3,2],ww[3,1,1],ww[3,1,2],ww[3,1,3]};
biases = {bb[2,1],bb[2,2],bb[2,3],bb[3,1]};
modelparameters = Join[weights, biases];


(* initialise model parameters *)
paramrules = Rule@@@Transpose@{modelparameters,RandomReal[{0,1}]&/@Range[Length[modelparameters]]}


data = {
  {{1,1},1},
  {{2,2},4},
  {{1.5,1.5},1.1},
  {{1.75,1.75},0.9}
};


sig[z_]:=1/(1+Exp[-z])

NNactivationstates = {
 {aa[1,1],aa[1,2]},
 {aa[2,1],aa[2,2],aa[2,3]},
 {aa[3,1]}
};

err[y_,yhat_]:=(y-yhat)^2


D[err[y,yhat],yhat]


(* run 1st point *)
pt = 1; 


NNlinwgtsums = NNactivationstates /. aa->zz;


(* step 1 *)
actstat0 = Rule@@@Transpose@{NNactivationstates[[1]],data[[pt,1]]}


(* step 2 *)
linwgt1 = Table[zz[2,i]->Sum[ww[2,i,j]*aa[1,j],{j,1,Length[NNactivationstates[[1]]]}]+bb[2,i],{i,1,Length[NNactivationstates[[2]]]}]
linwgt1 = linwgt1 /. paramrules /. actstat0
actstat1 = linwgt1 /. Rule[a_,b_]:>Rule[aa@@a,sig[b]]
linwgt2 = Table[zz[3,i]->Sum[ww[3,i,j]*aa[2,j],{j,1,Length[NNactivationstates[[2]]]}]+bb[3,i],{i,1,Length[NNactivationstates[[3]]]}]
linwgt2 = linwgt2 /. paramrules /. actstat1
actstat2 = linwgt2 /. Rule[a_,b_]:>Rule[aa@@a,sig[b]]


(* step 3 *)
NNerrorfuncs = NNactivationstates /. aa->Delta;
derr = D[err[data[[pt,2]],yhat],yhat] /. yhat->aa[3,1] /. actstat2
dsig = sig'[zz[3,1]] /. linwgt2
errfuncs2 = {NNerrorfuncs[[3,1]]->derr*dsig}


(* step 4 *)
errfuncs1 = Table[sig'[zz[2,j]]*Sum[Delta[3,i]*ww[3,i,j],{i,1,Length[NNactivationstates[[3]]]}],{j,1,Length[NNactivationstates[[2]]]}]
errfuncs1 = Rule@@@Transpose@{NNerrorfuncs[[2]],errfuncs1 /. paramrules /. linwgt1 /. actstat1 /. errfuncs2}


(* step 5 *)
dbias = Flatten@Table[dEb[l,i]->Delta[l,i],{l,2,3},{i,1,Length[NNactivationstates[[l]]]}] /. errfuncs1 /. errfuncs2
dwgt = Flatten@Table[dEw[l,i,j]->Delta[l,i]*aa[l-1,j],{l,2,3},{i,1,Length[NNactivationstates[[l]]]},{j,1,Length[NNactivationstates[[l-1]]]}] /. errfuncs1 /. errfuncs2 /. actstat0 /. actstat1 /. actstat2


dparams = Rule@@@Transpose@{d/@modelparameters,Flatten[Join[dwgt[[;;,2]],dbias[[;;,2]]]]}


(* Now put everything inside a function so we can loop over the training data *)
(* hard-code for single output architecture *)


BackPropagateGradient[NNactivationstates_,modelparameters_,sig_,err_,paramrules_,data_] := Module[{
nlayers,
NNlinwgtsums,
actstat,actstatrules,
linwgt,linwgtrules,
derr,dsig,
errfuncs,errfuncsrules,
NNerrorfuncs,
dbias,dwgt,dparams
},

nlayers = Length[NNactivationstates];

NNlinwgtsums = NNactivationstates /. aa->zz;
dparams = Rule@@@Transpose@{d/@modelparameters,0&/@modelparameters};

Do[
(* step 1 - initialise *)
actstat[0] = Rule@@@Transpose@{NNactivationstates[[1]],data[[pt,1]]};
actstatrules = actstat[0];
linwgtrules = {};

(* step 2 *)
Do[
  linwgt[ll-1] = Table[zz[ll,i]->Sum[ww[ll,i,j]*aa[ll-1,j],{j,1,Length[NNactivationstates[[ll-1]]]}]+bb[ll,i],{i,1,Length[NNactivationstates[[ll]]]}];
  linwgt[ll-1] = linwgt[ll-1] /. paramrules /. actstatrules;
  actstat[ll-1] = linwgt[ll-1] /. Rule[a_,b_]:>Rule[aa@@a,sig[b]];
  actstatrules = Join[actstatrules,actstat[ll-1]];
  linwgtrules = Join[linwgtrules,linwgt[ll-1]];
,{ll,2,nlayers}];

(* step 3 *)
NNerrorfuncs = NNactivationstates /. aa->Delta;
derr = D[err[data[[pt,2]],yhat],yhat] /. yhat->aa[nlayers,1] /. actstatrules;
dsig = sig'[zz[nlayers,1]] /. linwgtrules;
errfuncsrules = {NNerrorfuncs[[nlayers,1]]->derr*dsig};

(* step 4 *)
Do[
  errfuncs[nlayers-ll] = Table[sig'[zz[ll,j]]*Sum[Delta[ll+1,i]*ww[ll+1,i,j],{i,1,Length[NNactivationstates[[ll+1]]]}],{j,1,Length[NNactivationstates[[ll]]]}];
  errfuncs[nlayers-ll] = Rule@@@Transpose@{NNerrorfuncs[[2]],errfuncs[nlayers-ll] /. paramrules /. linwgtrules /. actstatrules /. errfuncsrules};
  errfuncsrules = Join[errfuncsrules,errfuncs[nlayers-ll]];
,{ll,nlayers-1,2,-1}];

(* step 5 *)
dbias = Flatten@Table[
  dEb[l,i]->Delta[l,i]
  ,{l,2,nlayers},{i,1,Length[NNactivationstates[[l]]]}] /. errfuncsrules;
dwgt = Flatten@Table[
  dEw[l,i,j]->Delta[l,i]*aa[l-1,j],{l,2,nlayers}
  ,{i,1,Length[NNactivationstates[[l]]]},{j,1,Length[NNactivationstates[[l-1]]]}] /. errfuncsrules /. actstatrules;

dparams = Rule@@@Transpose@{d/@modelparameters,dparams[[;;,2]]+Flatten[Join[dwgt[[;;,2]],dbias[[;;,2]]]]};

,{pt,1,Length[data]}];

(* finally average the gradient function *)
Return[dparams /. Rule[a_,b_]:>Rule[a,b/Length[data]]];
]


BackPropagateGradient[NNactivationstates,modelparameters,sig,err,paramrules,data]


TrainNN[NNactivationstates_,modelparameters_,sig_,err_,data_,iter_]:=Module[{dparams,paramrules,vv,eta},

paramrules = Rule@@@Transpose@{modelparameters,RandomReal[{-1,1}]&/@Range[Length[modelparameters]]};
Print["initialised parameters"];
Print["p0 = ",paramrules];
eta = 1;

Do[
dparams = BackPropagateGradient[NNactivationstates,modelparameters,sig,err,paramrules,data];
vv = dparams[[;;,2]];
paramrules = Rule@@@Transpose@{paramrules[[;;,1]],paramrules[[;;,2]]-vv*eta};
,{ii,1,iter}];

Return[paramrules];
];


trainedparams = TrainNN[NNactivationstates,modelparameters,sig,err,data,50]


NNeval[NNactivationstates_,sig_,trainedparams_,input_]:=Module[{
nlayers,
actstat,actstatrules,
linwgt,linwgtrules
},

nlayers = Length[NNactivationstates];

(* step 1 - initialise *)
actstat[0] = Rule@@@Transpose@{NNactivationstates[[1]],input};
actstatrules = actstat[0];
linwgtrules = {};

(* step 2 *)
Do[
  linwgt[ll-1] = Table[zz[ll,i]->Sum[ww[ll,i,j]*aa[ll-1,j],{j,1,Length[NNactivationstates[[ll-1]]]}]+bb[ll,i],{i,1,Length[NNactivationstates[[ll]]]}];
  linwgt[ll-1] = linwgt[ll-1] /. trainedparams /. actstatrules;
  actstat[ll-1] = linwgt[ll-1] /. Rule[a_,b_]:>Rule[aa@@a,sig[b]];
  actstatrules = Join[actstatrules,actstat[ll-1]];
  linwgtrules = Join[linwgtrules,linwgt[ll-1]];
,{ll,2,nlayers}];

Return[aa[nlayers,1] /. actstatrules];
]


trainedparams = TrainNN[NNactivationstates,modelparameters,sig,err,data,10]
NNeval[NNactivationstates,sig,trainedparams,#]&/@data[[;;,1]]-data[[;;,2]]

trainedparams = TrainNN[NNactivationstates,modelparameters,sig,err,data,50]
NNeval[NNactivationstates,sig,trainedparams,#]&/@data[[;;,1]]-data[[;;,2]]

trainedparams = TrainNN[NNactivationstates,modelparameters,sig,err,data,100]
NNeval[NNactivationstates,sig,trainedparams,#]&/@data[[;;,1]]-data[[;;,2]]


Plot3D[
  NNeval[NNactivationstates,sig,trainedparams,{x1,x2}]
,{x1,-2,2},{x2,-2,2}]


Plot3D[
  NNeval[NNactivationstates,sig,trainedparams,{x1,x2}]
,{x1,-2,2},{x2,-2,2}]


(* ::Subsection::Closed:: *)
(*Basic Training example*)


(* let's take a very simple surface and try to teach a neural network *)


FF[x1_,x2_]:=(Tanh[x1-x2]+1)/2


Plot3D[
  FF[x1,x2]
,{x1,-2,2},{x2,-2,2}]


(* we modify and improve the 'training' routine with features to track the improvements in the fit *)
(* we introduce both testing and training datasets to try and understand "cross-validation" *)

Options[TrainAndTestNN] = {"Initialisation"->0, "LearningRate"->1, "PrintChiSqInterval"->1};

TrainAndTestNN[NNactivationstates_,modelparameters_,sig_,err_,data_,testdata_,iter_,OptionsPattern[]]:=Module[{dparams,paramrules,parammatrix,vv,eta,Loss},

If[MatchQ[OptionValue["Initialisation"],0],
  paramrules = Rule@@@Transpose@{modelparameters,RandomReal[{-1,1}]&/@Range[Length[modelparameters]]};
,
  If[Length[OptionValue["Initialisation"]]!=Length[modelparameters],
    Print["Wrong format for inital parameters."];
    Return[$Fail];
  ];
  paramrules = Rule@@@Transpose@{modelparameters,OptionValue["Initialisation"]};
];
eta = OptionValue["LearningRate"];

(* store iterations of param values *)
parammatrix = {paramrules[[;;,2]]};

Do[

dparams = BackPropagateGradient[NNactivationstates,modelparameters,sig,err,paramrules,data];
vv = dparams[[;;,2]];

paramrules = Rule@@@Transpose@{paramrules[[;;,1]],paramrules[[;;,2]]-vv*eta};
parammatrix = Join[parammatrix,{paramrules[[;;,2]]-vv*eta}];

Loss = Sum[(NNeval[NNactivationstates,sig,paramrules,testdata[[i,1]]]-testdata[[i,2]])^2,{i,1,Length[testdata]}]/Length[testdata];
If[ii==1||Mod[ii,OptionValue["PrintChiSqInterval"]]==0,Print["iteration ",ii," Chi^2 = ", Loss]];

,{ii,1,iter}];

Return[{paramrules, parammatrix}];
];


(* generate data set *)
datapoints = RandomReal[{-2,2},{2}]&/@Range[800];
data = {#,FF@@#}&/@datapoints;

(* generate test data set *)
testdatapoints = RandomReal[{-2,2},{2}]&/@Range[200];
testdata = {#,FF@@#}&/@testdatapoints;


(* This plot is to demonstrate the test points conver the surface uniformly *)
fig1 = Plot3D[{
  FF[x1,x2]
},{x1,-2,2},{x2,-2,2}];
fig2 = ListPlot3D[Flatten/@data,{InterpolationOrder->0,PlotStyle->Blue}];
Show[fig1,fig2]


(* Set up a network as before with 13 parameters *)
(* We follow the notation that we introduced above *)
InitializeModel[depth_List]:=Module[{NNactivationstates,modelparameters},

NNactivationstates = Table[aa[l,d],{l,1,Length[depth]},{d,1,depth[[l]]}];

modelparameters = Flatten@Join[
  Table[ww[l,i,j],{l,2,Length[depth]},{i,1,depth[[l]]},{j,1,depth[[l-1]]}],
  Table[bb[l,i],{l,2,Length[depth]},{i,1,depth[[l]]}]
  ];

Print["model set up with ",Length[modelparameters]," parameters"];
Return[{NNactivationstates,modelparameters}];
];
(* recall he definitions of sig and err *)
sig[z_]:=1/(1+Exp[-z])
err[y_,yhat_]:=(y-yhat)^2


{myNN,modparams} = InitializeModel[{2,3,1}];
iterations = 30;
{trainedparams,paramevol} = TrainAndTestNN[myNN,modparams,sig,err,data,testdata,iterations];


ListPlot[Table[Transpose@{Range[iterations],paramevol[[2;;,i]]},{i,1,Length[paramevol[[1,;;]]]}],{Joined->True,PlotLabels->modparams}]


Plot3D[{
  FF[x1,x2],
  NNeval[myNN,sig,trainedparams,{x1,x2}]
},{x1,-2,2},{x2,-2,2}]


{ww[2,1,1]->-9.713111364707522`,ww[2,1,2]->8.938019606686034`,ww[3,1,1]->-4.872721834060293`,bb[2,1]->1.1822708759945577`,bb[3,1]->4.530331524568496`}


(* we can run this to consider the kind of surfaces we are generating as starting points for the fit *)
paramrules = Rule@@@Transpose@{modelparameters,RandomReal[{-10,10}]&/@Range[Length[modelparameters]]};
Plot3D[{
  FF[x1,x2],
  NNeval[NNactivationstates,sig,paramrules,{x1,x2}]
},{x1,-2,2},{x2,-2,2}]


(* maybe 13 parameters is still too many - let's try again with a simpler architecture  *)
{myNN,modparams} = InitializeModel[{2,2,1}];
iterations = 30;
{trainedparams,paramevol} = TrainAndTestNN[myNN,modparams,sig,err,data,testdata,iterations];

ListPlot[Table[Transpose@{Range[iterations],paramevol[[2;;,i]]},{i,1,Length[paramevol[[1,;;]]]}],{Joined->True,PlotLabels->modparams}]

Plot3D[{
  FF[x1,x2],
  NNeval[myNN,sig,trainedparams,{x1,x2}]
},{x1,-2,2},{x2,-2,2}]


iterations = 100;

{trainedparams,paramevol} = TrainAndTestNN[myNN,modparams,sig,err,data,testdata,iterations,{
  "PrintChiSqInterval"->10,
  "LearningRate"->3
  }];

ListPlot[Table[Transpose@{Range[iterations],paramevol[[2;;,i]]},{i,1,Length[paramevol[[1,;;]]]}],{Joined->True,PlotLabels->modparams}]

Plot3D[{
  FF[x1,x2],
  NNeval[myNN,sig,trainedparams,{x1,x2}]
},{x1,-2,2},{x2,-2,2}]


(* ::Subsection::Closed:: *)
(*Further improvements*)


(* we modify and improve the 'training' routine with features to track the improvements in the fit *)
(* we introduce both testing and training datasets to try and understand "cross-validation" *)

Options[TrainAndTestNNv2] = {"Initialisation"->0, "LearningRate"->1, "PrintChiSqInterval"->1, "EarlyStopping"->False}

TrainAndTestNNv2[NNactivationstates_,modelparameters_,sig_,err_,data_,testdata_,iter_,OptionsPattern[]]:=Module[{
  dparams,paramrules,parammatrix,
  vv,eta,
  LossTest,LossTrain,lossmatrix},

If[MatchQ[OptionValue["Initialisation"],0],
  paramrules = Rule@@@Transpose@{modelparameters,RandomReal[{-1,1}]&/@Range[Length[modelparameters]]};
,
  If[Length[OptionValue["Initialisation"]]!=Length[modelparameters],
    Print["Wrong format for inital parameters."];
    Return[$Fail];
  ];
  paramrules = Rule@@@Transpose@{modelparameters,OptionValue["Initialisation"]};
];
eta = OptionValue["LearningRate"];

(* store iterations of param values *)
parammatrix = {paramrules[[;;,2]]};
lossmatrix = {};

Do[

dparams = BackPropagateGradient[NNactivationstates,modelparameters,sig,err,paramrules,data];
vv = dparams[[;;,2]];

paramrules = Rule@@@Transpose@{paramrules[[;;,1]],paramrules[[;;,2]]-vv*eta};
parammatrix = Join[parammatrix,{paramrules[[;;,2]]-vv*eta}];

LossTrain = Sum[(NNeval[NNactivationstates,sig,paramrules,data[[i,1]]]-data[[i,2]])^2,{i,1,Length[data]}]/Length[data];
LossTest = Sum[(NNeval[NNactivationstates,sig,paramrules,testdata[[i,1]]]-testdata[[i,2]])^2,{i,1,Length[testdata]}]/Length[testdata];

If[ii==1||Mod[ii,OptionValue["PrintChiSqInterval"]]==0,Print["iteration ",ii," Chi^2[training] = ", LossTrain," Chi^2[testing] = ", LossTest]];

lossmatrix = Join[lossmatrix, {{LossTrain, LossTest}}];

If[OptionValue["EarlyStopping"]&&LossTest<LossTrain, Print["stopping early after ",ii," iterations"]; Continue[];];

,{ii,1,iter}];

Return[{paramrules, parammatrix, lossmatrix}];
];


FF[x1_,x2_]:=(Sin[x1]^2+Cos[x2]^2)/2


(* generate data set *)
datapoints = RandomReal[{-2,2},{2}]&/@Range[800];
data = {#,FF@@#}&/@datapoints;

(* generate test data set *)
testdatapoints = RandomReal[{-2,2},{2}]&/@Range[200];
testdata = {#,FF@@#}&/@testdatapoints;


Plot3D[
  FF[x1,x2]
,{x1,-2,2},{x2,-2,2}]


{myNN,modparams} = InitializeModel[{2,8,1}];
iterations = 50;

{trainedparams,paramevol,lossevol} = TrainAndTestNNv2[myNN,modparams,sig,err,data,testdata,iterations,{
  "PrintChiSqInterval"->10,
  "LearningRate"->3
  }];

ListPlot[Table[Transpose@{Range[iterations],paramevol[[2;;,i]]},{i,1,Length[paramevol[[1,;;]]]}],{Joined->True,PlotLabels->modparams}]

ListPlot[Transpose@lossevol,{PlotRange->{{0,0.2}},Joined->True}]

Plot3D[{
  FF[x1,x2],
  NNeval[myNN,sig,trainedparams,{x1,x2}]
},{x1,-2,2},{x2,-2,2}]


{myNN,modparams} = InitializeModel[{2,8,1}];
iterations = 1000;

{trainedparams,paramevol,lossevol} = TrainAndTestNNv2[myNN,modparams,sig,err,data,testdata,iterations,{
  "PrintChiSqInterval"->100,
  "LearningRate"->3,
  "EarlyStopping"->True
  }];

ListPlot[Table[Transpose@{Range[iterations],paramevol[[2;;,i]]},{i,1,Length[paramevol[[1,;;]]]}],{Joined->True,PlotLabels->modparams}]

ListPlot[Transpose@lossevol,{PlotRange->{{0,0.2}},Joined->True}]

Plot3D[{
  FF[x1,x2],
  NNeval[myNN,sig,trainedparams,{x1,x2}]
},{x1,-2,2},{x2,-2,2}]


(* ::Subsection::Closed:: *)
(*Machine learning functional forms with in-built routines*)


(* in-built Mathematica rountine... *)


p=Predict[Rule[#[[1]],#[[2]]]&/@data,Method->"NeuralNetwork"]


Plot3D[{FF[x1,x2],p[{x1,x2}]},{x1,-2,2},{x2,-2,2}]
