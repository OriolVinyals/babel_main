import os
import subprocess

def cmdSNR(audio_file, t_beg=None, t_end=None):
    if t_beg == None:
        cmd = '/u/vinyals/projects/swordfish/src/snreval/run_snreval_prj.sh ' + audio_file + ' -disp 0'
    else:
        cmd = '/u/vinyals/projects/swordfish/src/snreval/run_snreval_prj.sh ' + audio_file + ' '
        cmd += '-start ' + repr(t_beg) + ' -end ' + repr(t_end) + ' -disp 0'
    p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='/u/vinyals/projects/swordfish/src/snreval/')
    out, err = p.communicate()
    for line in out.split('\n'):
        if line.find('STNR')>-1:
            return float(line.split(' ')[3])
        
def cmdChunk(audio_file, input_string):
    cmd = 'iajoin ' + audio_file
    p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    p.communicate(input_string)
    
def cmdmrate(audio_file, t_beg=None, t_end=None):
    if t_beg == None:
        cmd = '/u/vinyals/projects/swordfish/src/mrate/src/get_mrate -i ' + audio_file
    else:
        cmd = '/u/vinyals/projects/swordfish/src/mrate/src/get_mrate -i ' + audio_file + ' '
        cmd += '-b ' + repr(t_beg) + ' -e ' + repr(t_end)
    p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    #print out.split('\n')[1]
    return float(out.split('\n')[1])

def cmdconvert(in_audio_file,out_audio_file,t_beg=None,t_end=None,format='MSWAVE'):
    #Note: we do convert samples to 16 bit encoding
    if t_beg == None:
        cmd = 'sndcat -f s -op ' + format + ' ' + in_audio_file + ' -o ' + out_audio_file
    else:
        cmd = 'sndcat -f s -op ' + format + ' -k ' + repr(t_beg) + ' -e ' + repr(t_end) + ' ' + in_audio_file + ' -o ' + out_audio_file
    p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    rc = p.returncode
    if(rc!=0):
        print out
        print err
    return