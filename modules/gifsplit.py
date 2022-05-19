!rm -rf gif_frames
!mkdir gif_frames

to_remove = []
to_add = []

for image_path in input_files:
  __, image_name = os.path.split(image_path)
  if image_name.endswith('.gif'):
    im = Image.open(image_path)
    for frame in range(0, im.n_frames):
      gif_frames_path = os.path.join("gif_frames", image_name + str(frame) + ".png")
      im.seek(frame)
      im.save(gif_frames_path)
      to_add.append(gif_frames_path)
    to_remove.append(image_path)

input_files = [image_path for image_path in input_files if image_path not in to_remove]
input_files.extend(to_add)
