(* ::Package:: *)

GetIsotropic[x_List]:=Module[{c,p,EE},
c = 2*x[[1]]-1;
p = 2*Pi*x[[2]];
EE = -Log[x[[3]]*x[[4]]];
Return[EE*{1, Sqrt[1-c^2]*Cos[p], Sqrt[1-c^2]*Sin[p], c}]
]

dot[x_,y_]:=x[[1]]*y[[1]]-(x[[2]]*y[[2]]+x[[3]]*y[[3]]+x[[4]]*y[[4]])

Boost[x_,gam_,A_,B_,phat_]:=Module[{p0,pvec},
p0 = x*(gam*phat[[1]]+B.phat[[2;;]]);
pvec = x*( phat[[2;;]] + B*phat[[1]] + A*B*(B.phat[[2;;]]) );
Return[Join[{p0},pvec]]
]


Options[RAMBO] = {"Debug"->False};
RAMBO[nfinal_,rts_,OptionsPattern[]]:=Module[{xx,phat,QQ,Q2,MM,A,B,gam,X,momsfinal,moms},

(* step 1 *)
xx = Table[RandomReal[{0,1},4],{i,1,nfinal}];
(* step 2 *)
phat = GetIsotropic/@xx;

If[OptionValue["Debug"],Print["on-shell final: ", dot[#,#]&/@phat]];

(* step 3 (boost) *)
QQ = Plus@@phat;
Q2 = dot[QQ,QQ];
MM = Sqrt[Q2];
B = -QQ[[2;;]]/MM;
X = rts/MM;
gam = QQ[[1]]/MM;
A = 1/(1+gam);

momsfinal = Boost[X,gam,A,B,#]&/@phat;

If[OptionValue["Debug"],Print["final mom conservation: ", Plus@@momsfinal]];

moms =Join[{
-rts/2*{1.,0,0,1.},
-rts/2*{1.,0,0,-1.}
},momsfinal];

If[OptionValue["Debug"],Print["mom conservation: ", Plus@@moms]];

Return[moms]

];


RAMBO[3,Sqrt[14],"Debug"->True]


(* example amplitude function - e+e\[Rule]qqg @ tree-level *)
(* mass regulated poles *)
TestAmp[{p1_,p2_,p3_,p4_,p5_}]:=Module[{mm,sab,s12,s23,s13,sa1,sa2,sa3,sb1,sb2,sb3},
  sab = dot[p1,p2];
  s12 = dot[p3,p4]; s13 = dot[p3,p5]; s23 = dot[p4,p5];
  sa1 = dot[p1,p3]; sa2 = dot[p1,p4]; sa3 = dot[p1,p5];
  sb1 = dot[p2,p3]; sb2 = dot[p2,p4]; sb3 = dot[p2,p5];
  Return[ (sa1^2+sa2^2+sb1^2+sb2^2)/(sab*s13*s23) ];
]


TestAmp@RAMBO[3,14]


MCintegrate[s_, nfinal_, amp_, iterations_]:=Module[{values},
values = {};
Do[
  values = Join[values,{amp@RAMBO[nfinal,Sqrt[s]]}];
,{ii,1,iterations}];

values = Table[(Plus@@(values[[1;;ii]]))/ii,{ii,1,iterations}];

Return[values];
];


results = MCintegrate[14,3,TestAmp, 10000];


ListPlot[results, {"Joined"->True}]


results[[1000]]
results[[10000]]


RAMBOn[N_,nfinal_,rts_,delta_]:=Module[{Ntrials,Ngenerated,pp,ppfinal,sep,moms},
 Ntrials = 0;
 Ngenerated = 0;
 moms = {};
 While[Ngenerated<N&&Ntrials<10^6,
  pp = RAMBO[nfinal,rts];
  ppfinal=pp[[3;;]];
  sep = Table[dot[ppfinal[[i]],ppfinal[[j]]]/rts^2,{i,1,nfinal-1},{j,i+1,nfinal}];
  If[Sort[Flatten@sep][[1]]>delta, Ngenerated+=1; moms = Join[moms,{pp}]];
  Ntrials += 1;
 ];
 Return[{Ntrials,moms}];
];


(* efficiency *)
{trials,moms} = RAMBOn[1000, 3, Sqrt[14], 0.01];
1000./trials
{trials,moms} = RAMBOn[1000, 3, Sqrt[14], 0.03];
1000./trials
{trials,moms} = RAMBOn[1000, 3, Sqrt[14], 0.1];
1000./trials


MCintegrate2[s_, nfinal_, amp_, iterations_, cut_]:=Module[{trials, moms, values},
{trials,moms} = RAMBOn[iterations, nfinal, Sqrt[s], cut];

values = {};
Do[
  values = Join[values,{amp@moms[[ii]]}];
,{ii,1,iterations}];

values = iterations/trials*Table[(Plus@@(values[[1;;ii]]))/ii,{ii,1,iterations}];

Return[{values, trials}];
];


{results, ntrials} = MCintegrate2[14,3,TestAmp, 100000, 0.01];


ListPlot[results[[1;;1000]], {"Joined"->True}]


ListPlot[results[[1000;;10000]], {"Joined"->True}]


ListPlot[results, {"Joined"->True}]


results[[1000]]
results[[10000]]
results[[100000]]


{results2, ntrials2} = MCintegrate2[14,3,TestAmp, 100000, 0.03];


100000./ntrials2
results2[[-1]]
ListPlot[results2, {"Joined"->True}]
results2[[1000]]
results2[[10000]]
results2[[100000]]


(* how do the amplitude evaluations contribute to the cross section? *)
{trials, moms} = RAMBOn[10000, 3, Sqrt[14], 0.001];
ampvals = TestAmp/@moms;
Histogram[Log[ampvals]]


(* 10 largest values *)
Sort[ampvals][[-10;;]]
pos = Position[ampvals,#][[1,1]]&/@%

Do[
Print[sep = Sort@Flatten@Table[dot[moms[[p,3;;]][[i]],moms[[p,3;;]][[j]]]/14,{i,1,3-1},{j,i+1,3}]];
,{p,pos}]


(* In order to improve efficiency, we need to weight phase space generator to prefer points which contribute a lot to the cross section *)


(* FKS splitting *)


PS = RAMBO[3,14]


(* all massless assumed *)
s[i_,j_,PS_]:=2*dot[PS[[i]],PS[[j]]]
S[i_,j_,D1_,PS_]:=1/s[i,j,PS]/D1


ShowFKSpartitions[nn_]:=Module[{DD},
DD = Sum[1/s[i,j],{i,3,nn},{j,i+1,nn}];
Return[Flatten@Table[1/s[i,j]/DD,{i,3,nn-1},{j,i+1,nn}]];
]


tmp = ShowFKSpartitions[5]//Factor
Normal@Series[tmp /. {s[3,4]->delta, s[3,5]->z*s[P,5], s[4,5]->(1-z)*s[P,5]},{delta,0,0}] (* 3||4 *)
Normal@Series[tmp /. {s[3,5]->delta, s[3,4]->z*s[P,4], s[4,5]->(1-z)*s[P,4]},{delta,0,0}]
Normal@Series[tmp /. {s[4,5]->delta, s[3,5]->z*s[P,3], s[3,4]->(1-z)*s[P,3]},{delta,0,0}]

Normal@Series[tmp /. {s[3,4]->delta, s[3,5]->delta},{delta,0,0}] (* p3 soft *)
Normal@Series[tmp /. {s[3,4]->delta, s[4,5]->delta},{delta,0,0}]
Normal@Series[tmp /. {s[3,5]->delta, s[4,5]->delta},{delta,0,0}]


tmp = ShowFKSpartitions[6]//Factor
Normal@Series[tmp /. {s[3,4]->delta, s[3,5]->z*s[P,5], s[4,5]->(1-z)*s[P,5]},{delta,0,0}] (* 3||4 *)
Normal@Series[tmp /. {s[3,5]->delta, s[3,4]->z*s[P,4], s[4,5]->(1-z)*s[P,4]},{delta,0,0}]
Normal@Series[tmp /. {s[4,5]->delta, s[3,5]->z*s[P,3], s[3,4]->(1-z)*s[P,3]},{delta,0,0}]

Normal@Series[tmp /. {s[3,4]->delta, s[3,5]->delta},{delta,0,0}] (* p3 soft *)
Normal@Series[tmp /. {s[3,4]->delta, s[4,5]->delta},{delta,0,0}]
Normal@Series[tmp /. {s[3,5]->delta, s[4,5]->delta},{delta,0,0}]


GetFKSpartitions[PS_]:=Module[{nn, DD},
nn = Length[PS];
DD = Sum[1/s[i,j,PS],{i,3,nn},{j,i+1,nn}];
Return[Flatten@Table[S[i,j,DD,PS],{i,3,nn-1},{j,i+1,nn}]];
]


GetFKSpartitions[PS]
Plus@@%


Do[
  Print[GetFKSpartitions[moms[[p]]]];
,{p,pos}]
