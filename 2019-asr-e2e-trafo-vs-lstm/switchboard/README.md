Raw Latex table comments:
```
% /u/bahar/workspace/asr/switchboard/20190612--transfo/config-train/trafo.12l.n512.ffdim4.pretrain.lr0008.ctc.dr0.2.config
% /work/smt3/bahar/expriments/asr/switchboard/20190612--transfo/data-recog/trafo.12l.n512.ffdim4.pretrain.lr0008.ctc.dr0.2.300
\quad Transformer & none & 16.0 & 30.5 & 22.4 \\

% /u/bahar/workspace/asr/switchboard/20190614--transfo/config-train/trafo.specaug4.12l.ffdim4.pretrain3a.lr0008.ctc.dr0.3.wu8.config
% /work/smt3/bahar/expriments/asr/switchboard/20190612--transfo/data-recog/trafo.specaug4.12l.ffdim4.pretrain3a.lr0008.ctc.dr0.3.wu8.288
\quad \quad + SpecAug &  & 	11.2 & 22.7	& 16.3 \\

% /u/bahar/workspace/asr/switchboard/20190614--transfo/config-train/trafo.specaug4.enc36l.dec12l.n512.ffdim2.pretrain.lr0008.ctc.dr0.4.wu.acc3.config
% /work/smt3/bahar/expriments/asr/switchboard/20190612--transfo/data-recog/trafo.specaug4.enc36l.dec12l.n512.ffdim2.pretrain.lr0008.ctc.dr0.4.wu.acc3.289
\quad \quad\quad + 36Enc &  & 	10.6 & 22.3 & 15.5\\

%|| 17.8 || 72 || base.6l.red6.pretrain-start4l-red6-below-grow3 || 0:26h || 31:37h ||  ||
\quad LSTM & none &  11.9 & 23.7 & 17.7 \\

% || 15.7 || 160 || base2.specaug4 || 0:28h || 75:17h ||  ||
\quad \quad + SpecAug &  & \phantom{0}\textbf{9.9} & 21.5 & \textbf{14.7} \\

% || 15.9 || 178 || base2.smlp2.specaug4 || 0:27h || 79:19h ||  ||
\quad \quad\quad + ExpV &  & 10.4 & 21.4 & 15.1 \\

% || 15.4 || 198 || base2.conv2l.specaug4 || 0:28h || 91:51h ||  ||
\quad \quad\quad + Conv &  & 10.1 & \textbf{20.6} & \textbf{14.7} \\
 
% (irie/setups) base2.conv2l.specaug4a.trafo_lm_fusion.tune_eos.160 
\quad \quad\quad\quad + LM-EOS & Trafo & \phantom{0}\textbf{9.3} & \textbf{20.3} & \textbf{14.2} \\
```
The commented number (17.8, 15.7, 15.9, 15.4) is the WER on Hub5'00 (SWB/CH combined).
