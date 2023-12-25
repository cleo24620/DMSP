function [Vht, Vht0, aVht] = dHTA(V, B, time)
% arguments in:
%  V ---- observed plasma velocity in GSE coordinates (km/s )
%  B ---- observed mangetic field in GSE coordinates (nT)
%  time ---- time intervals of the boservations(s)
% arguments out:
%  Vht ---- velocity of the HT frame (km/s)
%  Vht0 ---- initial velocity of the accelerated HT frame (km/s)
%  aVht ---- acceleration of the accelerated HT frame(km/s^2)
% see also: xxx

% Designed by  Huijun, Li , spaceweather lab. of CSSAR of CAS
% 21 Dec, 2006, created 

% check the out put parameter list
if nargout>3
    error('too many output arguments.'); 
end

% check the input parameter list
if nargin ~= 3
    error('wrong number of iutput arguments.');
end

% function body
% Now determin HT velocity for both VHT=Const and
% the accelerating HT frame: VHT=VHT0+aHT*time 
r=length(time);
for m=1:r
	BB=B(m,1)^2+B(m,2)^2+B(m,3)^2;
	for i=1:3
		for j=1:3
			if i==j
				Kijm(i,j,m)=BB*(1-B(m,i)*B(m,j)/BB);
			else
				Kijm(i,j,m)=-B(m,i)*B(m,j);
			end
		end
	end
	K1ijm(:,:,m)=Kijm(:,:,m)*time(m);
	K2ijm(:,:,m)=Kijm(:,:,m)*(time(m)^2);
end

K0=sum(Kijm,3)/r;
K1=sum(K1ijm,3)/r;
K2=sum(K2ijm,3)/r;

for m=1:r
	Rb(:,m)=Kijm(:,:,m)*([V(m,1) V(m,2) V(m,3)]');
	Rb2(:,m)=Rb(:,m)*time(m);
end
%
VHT=inv(K0)*(sum(Rb, 2)/r);
RHS1=sum(Rb,2)/r;
RHS2=sum(Rb2,2)/r;
VHT0aHT=([K0 K1;K1 K2])\([RHS1;RHS2]);
VHT0=VHT0aHT(1:3);
aHT=VHT0aHT(4:6);
%
Vht = VHT;
Vht0 = VHT0; %save aVHT aVHT;
aVht = aHT; %save aHT aHT;

