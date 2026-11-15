import subprocess, sys, json
C="/private/tmp/claude-501/c/"
segs=json.loads(sys.argv[1])   # [[clip, ss, to], ...]
X=0.6
inp=[]; filt=[]; 
for i,(c,ss,to) in enumerate(segs):
    inp+=["-i",C+"clips/"+c+".mp4"]
    filt.append(f"[{i}:v]trim={ss}:{to},setpts=PTS-STARTPTS,fps=25,format=yuv420p[v{i}]")
n=len(segs)
inp+=["-loop","1","-t","3.5","-framerate","25","-i",C+"endcard.png"]
filt.append(f"[{n}:v]fps=25,format=yuv420p,fade=t=in:st=0:d=0.8[v{n}]")
lens=[to-ss for _,ss,to in segs]+[3.5]
cur="v0"; t=lens[0]
for i in range(1,n+1):
    off=t-X
    filt.append(f"[{cur}][v{i}]xfade=transition=fade:duration={X}:offset={off:.3f}[x{i}]")
    cur=f"x{i}"; t=off+lens[i]
total=t
filt.append(f"[{cur}]format=yuv420p[vout]")
fc=";".join(filt)
def run(out, extra_in, extra_map, vf=None):
    f=fc if not vf else fc.replace("[vout]","[vtmp];[vtmp]"+vf+"[vout]")
    cmd=["ffmpeg","-v","error","-y"]+inp+extra_in+["-filter_complex",f,"-map","[vout]"]+extra_map+["-c:v","libx264","-crf","20","-preset","medium","-pix_fmt","yuv420p","-movflags","+faststart",out]
    subprocess.run(cmd,check=True)
print("total",total)
aud=["-i",C+"rap.m4a"]
am=["-map",f"{n+1}:a","-af",f"afade=t=out:st=52:d=3,apad","-c:a","aac","-b:a","160k","-t",f"{total:.2f}"]
O="~/Downloads/"
run(O+"cinematic_demo.mp4",[],[],None)
run(O+"cinematic_demo_rap.mp4",aud,am,None)
run(O+"cinematic_demo_vertical.mp4",aud,am,"crop=608:1080:656:0,scale=1080:1920:flags=lanczos")
