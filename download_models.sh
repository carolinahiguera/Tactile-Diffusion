# tactile diffusion trained with YCB-Slide dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1atrtbLuOonA5lf8GIssLcoDwL7P-cQH6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1atrtbLuOonA5lf8GIssLcoDwL7P-cQH6" -O difusion_ycb_slide.pt && rm -rf /tmp/cookies.txt
mkdir -p ./tacto_diffusion/outputs/model_ycb_slide/checkpoints/
mv difusion_ycb_slide.pt ./tacto_diffusion/outputs/model_ycb_slide/checkpoints/
echo "Downloaded diffusion model trained with YCB-Slide dataset"

# tactile diffusion fine tuned with Braille dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NCX65UWSF1LxRd13QXUlVY-ilb4snnCN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NCX65UWSF1LxRd13QXUlVY-ilb4snnCN" -O difusion_braille.pt && rm -rf /tmp/cookies.txt
mkdir -p ./tacto_diffusion/outputs/model_braille/checkpoints/
mv difusion_braille.pt ./tacto_diffusion/outputs/model_braille/checkpoints/
echo "Downloaded diffusion model fine-tuned with Braille dataset"

# classifier trained with sim data
mkdir -p ./braille_clf/checkpoints/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k_FQLmAuBwXTO8yyHaYfglTzEyji4uJz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1k_FQLmAuBwXTO8yyHaYfglTzEyji4uJz" -O clf_tacto.pth && rm -rf /tmp/cookies.txt
mv clf_tacto.pth ./braille_clf/checkpoints/
echo "Downloaded classifier trained with sim data"

# classifier trained with tacto+fine-tuning
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hVGDBpFVynhWwGpI9ko4gXcs84IHo8SY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hVGDBpFVynhWwGpI9ko4gXcs84IHo8SY" -O clf_tacto_finetune_0.8.pth && rm -rf /tmp/cookies.txt
mv clf_tacto_finetune_0.8.pth ./braille_clf/checkpoints/
echo "Downloaded classifier trained with tacto+fine-tuning"

# classifier trained with tacto_diffusion data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15dcguWSkrQ23JgvuV3CwRVKKAky_tDVN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15dcguWSkrQ23JgvuV3CwRVKKAky_tDVN" -O clf_diffusion.pth && rm -rf /tmp/cookies.txt
mv clf_diffusion.pth ./braille_clf/checkpoints/
echo "Downloaded classifier trained with tacto_diffusion data"