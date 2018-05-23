close all;
clear all

[hdr{1}, record{1}] = edfread('/home/asus/MyProjects/Happiness_BCI/baseline/NeoRec_2018-02-05_13-25-25.edf');
[hdr{2}, record{2}] = edfread('/home/asus/MyProjects/Happiness_BCI/meditation/NeoRec_2018-02-05_13-51-12.edf');
Fs = 1000;
[b,a] = butter(3,[2 50]/(Fs/2));

for r = 1:2
    x{r} = record{r}(1:32,:);
    xf{r} = filtfilt(b,a,x{r}')';
end;
range_bl =[10.1*60*Fs:14.9*60*Fs];
range_md =[11.1*60*Fs:15.9*60*Fs];

[W,S] = runica([xf{1}(:,range_bl) xf{2}(:,range_md)]);
xx = [xf{1}(:,range_bl) xf{2}(:,range_md)];
Q = W*S;
z = Q*xx;
close all
plotmatr(z(:,1:5:end))
imagesc(z(:,1:7000))
[Pzz,F] = pwelch(z(:,1:7000)',1*Fs,0.5*Fs,1*Fs,Fs);

% visual inspection
rej_ind = [3,32]
iQ = inv(Q);
z_cl = z;
z_cl(rej_ind,:) = 0;
x_cl = iQ*z_cl;

figure
plotmatr(x_cl(:,1:30000));

[Pxx_bl,F] = pwelch(x_cl(:,1:length(range_bl))',1*Fs,0.5*Fs,5*Fs,Fs);
[Pxx_md,F] = pwelch(x_cl(:,length(range_bl)+1:length(range_bl)+length(range_md))',1*Fs,0.5*Fs,5*Fs,Fs);

figure
h = imagesc(F(1:500),1:32,Pxx_md(1:500,:)'-Pxx_bl(1:500,:)');
set(h.Parent,'Ytick',1:32);
set(h.Parent,'YtickLabel',hdr{1}.label(1:32));

range = 1:Fs;
x_bl = x_cl(:,1:length(range_bl));
for w = 1:2*fix(length(range_bl)/Fs)-1
    PG_bl(:,:,w) = fft(x_bl(:,range)')';
    range = range+fix(Fs/2);
end;

range = 1:Fs;
x_md = x_cl(:,length(range_bl)+1:length(range_bl)+length(range_md));
for w = 1:2*fix(length(range_md)/Fs)-1
    PG_md(:,:,w) = fft(x_md(:,range)')';
    range = range+fix(Fs/2);
end;

ind_mdbl = 1:(size(PG_md,3)+size(PG_bl,3));
clear PG_sg;
PG_mdbl = cat(3,PG_md,PG_bl);
clear PG_md_sg;
for mc = 1:100
    ind_mdbl_p = ind_mdbl(randperm(size(PG_mdbl,3)));
    PG_md_sg(:,:,mc) = mean(abs(cat(3,PG_mdbl(:,:,ind_mdbl_p(1:fix(end/2))))),3);
    PG_bl_sg(:,:,mc) = mean(abs(cat(3,PG_mdbl(:,:,ind_mdbl_p(fix(end/2)+1:end)))),3);
end
PG_sg = PG_md_sg-PG_bl_sg;






close all
tmp = zscore(:,1:300).*(abs(zscore(:,1:300))>1.0);
tmp1 = tmp;
tmp1(1,end) = -max(abs(tmp(:)));
tmp1(end,end) = max(abs(tmp(:)));

h = imagesc(F(1:300),1:32,tmp1);
set(h.Parent,'Ytick',1:32);
set(h.Parent,'YtickLabel',hdr{1}.label(1:32));
colormap(hsv)
xlabel('Frequency');
title('Randomization test Z-score')
colorbar
colormap(jet)


PSD_sg_av =mean(PG_sg,3);
PSD_sg_std =std(PG_sg,[],3); 
PSD_md_av =mean(abs(PG_md),3); 
PSD_bl_av =mean(abs(PG_bl),3); 
PSD_mdbl_av = PSD_md_av-PSD_bl_av;
zscore = (PSD_mdbl_av-PSD_sg_av)./(PSD_sg_std)

close all
tmp = zscore(:,1:300).*(abs(zscore(:,1:300))>1.0);
tmp1 = tmp;
tmp1(1,end) = -max(abs(tmp(:)));
tmp1(end,end) = max(abs(tmp(:)));

h = imagesc(F(1:300),1:32,tmp1);
set(h.Parent,'Ytick',1:32);
set(h.Parent,'YtickLabel',hdr{1}.label(1:32));
colormap(hsv)
xlabel('Frequency');
title('Randomization test Z-score')
colorbar
colormap(jet)
figure
imagesc(zscore(:,1:300))
