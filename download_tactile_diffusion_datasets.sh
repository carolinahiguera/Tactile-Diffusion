wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UhL9QixREHWPklBLFft-j7xDFYIs9Zty' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UhL9QixREHWPklBLFft-j7xDFYIs9Zty" -O datasets.tar.xz && rm -rf /tmp/cookies.txt
tar -xf datasets.tar.xz
rm datasets.tar.xz
echo "Downloaded datasets for Tactile Diffusion (a sample of YCB-Slide + Braille datasets)"