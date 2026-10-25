"""Real-time LEDMatrix audio visualizer using Raspberry Pi ALSA arecord."""
from __future__ import annotations
import math, re, shutil, subprocess, threading, time
import numpy as np
from PIL import Image, ImageDraw
from src.plugin_system.base_plugin import BasePlugin

class AudioVisualizerPlugin(BasePlugin):
    needs_high_fps = True

    def __init__(self, plugin_id, config, display_manager, cache_manager, plugin_manager):
        super().__init__(plugin_id, config, display_manager, cache_manager, plugin_manager)
        self._lock=threading.Lock(); self._process=None; self._reader=None; self._stop_event=threading.Event()
        self._audio_buffer=np.zeros(8192,dtype=np.float32); self._buffer_write=0; self._buffer_filled=0
        self._last_audio_active_at=float('-inf'); self._current_rms=0.0; self._audio_error=None
        self._levels=np.zeros(16,dtype=np.float32); self._peaks=np.zeros(16,dtype=np.float32)
        self._needle_level=0.0; self._vu_reference=0.0
        self._load_config(config); self._resize_state(); self._start_audio()

    def _load_config(self,c):
        self.input_device=c.get('input_device'); self.vu_mode=str(c.get('vu_mode','spectrum')); self.sensitivity=float(c.get('sensitivity',8.0)); self.hold_while_audio_active=bool(c.get('hold_while_audio_active',False)); self.activity_threshold=float(c.get('activity_threshold',0.008)); self.release_delay_seconds=float(c.get('release_delay_seconds',2.5)); self.sample_rate=int(c.get('sample_rate',44100)); self.fft_size=int(c.get('fft_size',2048)); self.min_frequency=float(c.get('min_frequency',60)); self.max_frequency=float(c.get('max_frequency',16000)); self.bar_count=int(c.get('bar_count',16)); self.smoothing=float(c.get('smoothing',0.55)); self.peak_decay=float(c.get('peak_decay',0.8)); self.show_peaks=bool(c.get('show_peaks',True)); self.color_low=tuple(c.get('color_low',[0,255,80])); self.color_high=tuple(c.get('color_high',[255,30,0])); self.display_duration=float(c.get('display_duration',15))

    def _resize_state(self):
        if len(self._levels)!=self.bar_count:self._levels=np.zeros(self.bar_count,dtype=np.float32);self._peaks=np.zeros(self.bar_count,dtype=np.float32)
        wanted=max(self.fft_size*4,4096)
        with self._lock:
            if len(self._audio_buffer)!=wanted:self._audio_buffer=np.zeros(wanted,dtype=np.float32);self._buffer_write=self._buffer_filled=0

    def on_config_change(self,new_config):
        old_audio=(self.input_device,self.sample_rate,self.fft_size); old_mode=self.vu_mode
        super().on_config_change(new_config)
        # Use the callback payload directly. Some LEDMatrix versions notify the plugin
        # before self.config has been refreshed, which otherwise leaves dropdown changes stale.
        self.config=new_config
        self._load_config(new_config); self._resize_state()
        if old_mode!=self.vu_mode:
            self._needle_level=0.0; self._vu_reference=0.0
            self.logger.info('VU mode changed from %s to %s',old_mode,self.vu_mode)
        if old_audio!=(self.input_device,self.sample_rate,self.fft_size):self._stop_audio();self._start_audio()
    def get_display_duration(self):return self.display_duration
    def has_live_priority(self):return self.hold_while_audio_active
    def has_live_content(self):
        if not self.hold_while_audio_active or not self._capture_running():return False
        with self._lock:last=self._last_audio_active_at
        return time.monotonic()-last<=self.release_delay_seconds
    def get_live_modes(self):return ['audio-visualizer']
    def _capture_running(self):return self._process is not None and self._process.poll() is None

    def _device(self):
        value=self.input_device
        if value not in (None,'','default','auto'):
            text=str(value).strip();return 'plughw:'+text+',0' if text.isdigit() else text
        try:
            result=subprocess.run(['/usr/bin/arecord','-l'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=3,check=False)
            match=re.search(r'^card\s+(\d+):.*?device\s+(\d+):',result.stdout,re.MULTILINE)
            if match:return 'plughw:{},{}'.format(match.group(1),match.group(2))
        except Exception as exc:self.logger.warning('Could not auto-detect microphone: %s',exc)
        return 'default'

    def _start_audio(self):
        if self._capture_running():return
        arecord=shutil.which('arecord')
        if not arecord:self._audio_error='arecord is unavailable';self.logger.error(self._audio_error);return
        device=self._device();command=[arecord,'-q','-D',device,'-f','S16_LE','-c','1','-r',str(self.sample_rate),'-t','raw']
        try:
            self._stop_event.clear();self._process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0);self._reader=threading.Thread(target=self._read_audio,daemon=True,name='audio-visualizer-capture');self._reader.start();self._audio_error=None;self.logger.info('Audio capture started using ALSA device %s',device)
        except Exception as exc:self._process=None;self._audio_error=str(exc);self.logger.error('Could not start ALSA audio capture: %s',exc,exc_info=True)

    def _read_audio(self):
        process=self._process
        try:
            while not self._stop_event.is_set() and process and process.poll() is None:
                data=process.stdout.read(1024) if process.stdout else b''
                if not data:break
                samples=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768.0
                if samples.size:self._accept_samples(samples)
        except Exception as exc:
            if not self._stop_event.is_set():self._audio_error=str(exc);self.logger.error('Audio reader failed: %s',exc,exc_info=True)
        if process and process.poll() is not None and not self._stop_event.is_set():
            try:error=process.stderr.read().decode(errors='replace').strip() if process.stderr else ''
            except Exception:error=''
            self._audio_error=error or 'arecord exited with code {}'.format(process.returncode);self.logger.error('Audio capture stopped: %s',self._audio_error)

    def _accept_samples(self,samples):
        rms=float(np.sqrt(np.mean(samples*samples)));now=time.monotonic()
        with self._lock:
            self._current_rms=rms
            if rms>=self.activity_threshold:self._last_audio_active_at=now
            cap=len(self._audio_buffer);count=min(samples.size,cap);samples=samples[-count:];end=self._buffer_write+count
            if end<=cap:self._audio_buffer[self._buffer_write:end]=samples
            else:
                first=cap-self._buffer_write;self._audio_buffer[self._buffer_write:]=samples[:first];self._audio_buffer[:count-first]=samples[first:]
            self._buffer_write=(self._buffer_write+count)%cap;self._buffer_filled=min(cap,self._buffer_filled+count)

    def _stop_audio(self):
        self._stop_event.set();process,self._process=self._process,None
        if process:
            try:process.terminate();process.wait(timeout=1)
            except Exception:
                try:process.kill()
                except Exception:pass
        reader,self._reader=self._reader,None
        if reader and reader.is_alive():reader.join(timeout=1)
    def cleanup(self):self._stop_audio();super().cleanup()
    def update(self):
        if not self._capture_running():self._start_audio()

    def _latest_samples(self):
        with self._lock:
            n=min(self._buffer_filled,self.fft_size);result=np.zeros(self.fft_size,dtype=np.float32)
            if n<=0:return result
            output=self.fft_size-n;start=(self._buffer_write-n)%len(self._audio_buffer);end=start+n
            if end<=len(self._audio_buffer):result[output:]=self._audio_buffer[start:end]
            else:
                first=len(self._audio_buffer)-start;result[output:output+first]=self._audio_buffer[start:];result[output+first:]=self._audio_buffer[:n-first]
            return result

    def _spectrum(self):
        samples=self._latest_samples();samples-=float(np.mean(samples));spectrum=np.abs(np.fft.rfft(samples*np.hanning(len(samples))))/max(1.0,len(samples)/2);freqs=np.fft.rfftfreq(len(samples),d=1.0/self.sample_rate);nyquist=self.sample_rate/2;low=max(20,min(self.min_frequency,nyquist-1));high=max(low+1,min(self.max_frequency,nyquist));edges=np.geomspace(low,high,self.bar_count+1);raw=np.zeros(self.bar_count,dtype=np.float32)
        for i in range(self.bar_count):
            mask=(freqs>=edges[i])&(freqs<edges[i+1])
            if np.any(mask):values=spectrum[mask];raw[i]=float(np.sqrt(np.mean(values*values)))
        db=20*np.log10(np.maximum(raw,1e-8))+20*math.log10(max(.01,self.sensitivity));target=np.clip((db+80)/80,0,1).astype(np.float32);alpha=np.clip(self.smoothing,0,.95);self._levels=alpha*self._levels+(1-alpha)*target;return self._levels

    @staticmethod
    def _color(a,b,t):t=max(0,min(1,float(t)));return tuple(int(x+(y-x)*t) for x,y in zip(a,b))

    def _render_spectrum(self,image,draw,width,height):
        levels=self._spectrum();usable=max(1,height-1);slot=width/float(self.bar_count)
        for i,level in enumerate(levels):
            bar=max(0,min(usable,int(round(level*usable))));x0=int(round(i*slot));x1=max(x0,int(round((i+1)*slot))-1)
            if x1>x0:x1-=1
            for y in range(bar):draw.line((x0,height-1-y,x1,height-1-y),fill=self._color(self.color_low,self.color_high,y/max(1,usable-1)))
            if self.show_peaks:
                self._peaks[i]=float(bar) if bar>=self._peaks[i] else max(0,self._peaks[i]-self.peak_decay);py=height-1-int(round(self._peaks[i]))
                if 0<=py<height:draw.line((x0,py,x1,py),fill=self.color_high)

    def _render_analog(self,image,draw,width,height):
        with self._lock:rms=self._current_rms
        signal=max(rms*max(0.01,self.sensitivity),1e-7)
        if self._vu_reference<=0.0:self._vu_reference=max(signal,1e-5)
        if signal>self._vu_reference:self._vu_reference += (signal-self._vu_reference)*0.35
        else:self._vu_reference=max(1e-5,self._vu_reference*0.9995)
        relative_db=20.0*math.log10(max(signal,1e-7)/max(self._vu_reference,1e-7));vu_db=max(-20.0,min(3.0,relative_db))
        scale=[(-20.0,0.00),(-10.0,0.25),(-7.0,0.39),(-5.0,0.50),(-3.0,0.62),(0.0,0.82),(3.0,1.00)];level=0.0
        for i in range(len(scale)-1):
            db0,p0=scale[i];db1,p1=scale[i+1]
            if vu_db<=db1:
                fraction=(vu_db-db0)/(db1-db0) if db1!=db0 else 0.0;level=p0+(p1-p0)*max(0.0,min(1.0,fraction));break
        else:level=1.0
        rate=0.58 if level>self._needle_level else 0.14;self._needle_level += (level-self._needle_level)*rate
        cx=width//2; cy=height+10; radius=max(18,min(width//2-3,height+7));start=math.radians(205);end=math.radians(335)
        def point(angle,r):return (int(round(cx+math.cos(angle)*r)),int(round(cy+math.sin(angle)*r)))
        arc_box=(cx-radius,cy-radius,cx+radius,cy+radius);draw.arc(arc_box,start=205,end=335,fill=(100,100,60),width=1)
        labels=[(-20,0.00),(-10,0.25),(-7,0.39),(-5,0.50),(-3,0.62),(0,0.82),(3,1.00)]
        for db,pos in labels:
            a=start+(end-start)*pos;outer=point(a,radius);inner=point(a,radius-4);color=(255,50,20) if db>=0 else ((255,190,20) if db>=-5 else (80,220,80));draw.line((inner[0],inner[1],outer[0],outer[1]),fill=color);text=str(db) if db<0 else ('0' if db==0 else '+3');tx,ty=point(a,radius-9);draw.text((tx-len(text)*3,ty-3),text,fill=color)
        draw.text((max(0,cx-6),0),'VU',fill=(255,255,255));angle=start+(end-start)*self._needle_level;tip=point(angle,radius-2);pivot=(cx,cy);draw.line((pivot[0],pivot[1],tip[0],tip[1]),fill=(255,40,20),width=2);draw.ellipse((cx-2,height-3,cx+2,height+1),fill=(220,180,80))

    def _render(self):
        width,height=int(self.display_manager.width),int(self.display_manager.height);image=Image.new('RGB',(width,height),(0,0,0));draw=ImageDraw.Draw(image)
        if not self._capture_running():draw.rectangle((0,0,width-1,height-1),outline=(100,0,0));draw.line((0,0,width-1,height-1),fill=(255,0,0));draw.line((width-1,0,0,height-1),fill=(255,0,0));return image
        if self.vu_mode=='analog_needle':self._render_analog(image,draw,width,height)
        else:self._render_spectrum(image,draw,width,height)
        return image

    def display(self,force_clear=False):
        try:self.display_manager.image=self._render();self.display_manager.draw=ImageDraw.Draw(self.display_manager.image);self.display_manager.update_display()
        except Exception as exc:self.logger.error('Audio visualizer render failed: %s',exc,exc_info=True)
    def validate_config(self):return super().validate_config() and self.min_frequency<self.max_frequency
    def get_info(self):
        info=super().get_info()
        with self._lock:rms=self._current_rms
        info.update({'audio_input_running':self._capture_running(),'audio_error':self._audio_error,'rms_level':round(rms,5),'audio_active':self.has_live_content(),'vu_mode':self.vu_mode,'input_device':self.input_device or 'auto','sample_rate':self.sample_rate,'fft_size':self.fft_size});return info
