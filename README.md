I build this node to have more control over the character placement and the possibility of easily creating images with more than two characters.

This node lets you give each character their own prompt and their own mask (any shape), plus one background prompt for the whole scene — and blends them together inside the AI's image-generation process itself.

Where the character masks don't overlap, each character is generated almost entirely from their own prompt. Where masks do overlap, both prompts blend smoothly and proportionally right there — instead of the model getting confused by multiple prompts stapled together. If one character dominates too much, play with the mask-strength.

Background conditioning:
Make sure the dimensions match the size of the latent image. The mask for the background is automatically defined inside the node and fills the whole image. To make it easier, I’ve also added my Fx Smart Latent Image node. Just connect the width / height output into the Fx Character Attention input and the background mask will always match the latent image. The strength of the background conditioning inside the character mask areas can be changed (background_bleed).

Please note: You probably won’t be able to create a perfect and clear image with one generation, especially when using ControlNet. I use inpainting afterwards to make the image look good. It is a long process, because I inpaint every detail of the image again.

About LoRAs:
A LoRA will affect every condition and is not limited to the character prompt.
