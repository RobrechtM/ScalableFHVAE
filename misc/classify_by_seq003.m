%classify_by_seq003
% dialect/gender classification on cogen
groundtruth='/users/spraak/hvanhamm/repos/ScalableFHVAE/misc/cogen_test.fac';
%indir='/esat/spchtemp/scratch/hvanhamm/fhvae_timit/exp/cgn_per_speaker_afgklno/reg_fhvae_hs_e116_s5000_p10_a10.0_b1000.0_n2001_e0.1/txt_cogen/';
%indir='/esat/spchtemp/scratch/hvanhamm/fhvae_timit/exp/cgn_per_speaker_afgklno/reg_fhvae_e115_s5000_p10_a10.0_b10.0_e0.1/txt/';
indir = '/esat/spchtemp/scratch/hvanhamm/fhvae_timit/exp/cgn_per_speaker/nogau_reg_fhvae_e99_s5000_p10_a10.0_b1.0_c1.0_e0.01/txt/';
% classification rate to examine
key={'gender','reg1'};
len=[2 4]; % number of classes for each key

fid = fopen(groundtruth);
header=split(fgetl(fid));
header{1}=strrep(header{1},'#','');
fmt = repmat('%s ',1,length(header));
gt=textscan(fid,fmt);
fclose(fid);
% S=struct();
% for k=1:length(header)
%     S=setfield(S,header{k},gt{k});
% end
disp(indir)
for i = 1:length(key)
    fid=fopen([indir key{i} '.scp']);
    fmt = ['%s [ ' repmat('%f ',1,len(i)) ']'];
    indata = textscan(fid,fmt);
    fclose(fid);
    probs=[indata{2:end}];
    [pr,ii]=max(probs,[],2);
    [~,k]=ismember(key{i},header);
    jj=str2double(regexprep(gt{k},'[a-z,A-Z]','')); % all ground truth
    [~,seq_nr]=ismember(indata{1},gt{1});
    truth = jj(seq_nr);
    correct = sum(ii == truth);
    fprintf('%s per sequence %5.2f\n',key{i},100*correct/length(ii));
    % group sequences per speaker
    spk='';
    votes=[];
    S=struct('spk','','votes',[],'gt',[]);
    for s = 1:length(indata{1}),
        ss = split(indata{1}{s},'_');
        if ~strcmp(ss{1},spk),
            % new speaker
            if ~isempty(votes) % add speaker to archive
                S(end+1).spk=spk;
                S(end).votes=votes;
                S(end).gt=truth(s-1);
                S(end).majority=mode(votes);
            end
            spk=ss{1};
            votes=[];
        end
        votes=[votes ii(s)];
    end
    correct = sum([S.majority] == [S.gt]);
    fprintf('%s per speaker %5.2f\n',key{i},100*correct/length(S));
    
end