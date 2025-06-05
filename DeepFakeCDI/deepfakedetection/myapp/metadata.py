import os
import sys
import json
import datetime
import tempfile
import urllib.request
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from pathlib import Path

# For local file metadata
try:
    from hachoir.parser import createParser
    from hachoir.metadata import extractMetadata
    from pymediainfo import MediaInfo
    import ffmpeg
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    print("Installing required packages...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "hachoir", "pymediainfo", "ffmpeg-python", "pillow"])
    from hachoir.parser import createParser
    from hachoir.metadata import extractMetadata
    from pymediainfo import MediaInfo
    import ffmpeg
    from PIL import Image
    from PIL.ExifTags import TAGS

# For YouTube video metadata
try:
    from pytube import YouTube
except ImportError:
    print("Installing pytube...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytube"])
    from pytube import YouTube


class VideoMetadataExtractor:
    def __init__(self):
        self.metadata = {}
        self.temp_dir = tempfile.mkdtemp()

    def extract_from_local_file(self, file_path):
        """Extract metadata from a local video file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.metadata['filename'] = os.path.basename(file_path)
        self.metadata['file_path'] = os.path.abspath(file_path)
        self.metadata['file_size'] = os.path.getsize(file_path)
        self.metadata['file_size_human'] = self._human_readable_size(self.metadata['file_size'])
        self.metadata['last_modified'] = datetime.datetime.fromtimestamp(
            os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['creation_time'] = datetime.datetime.fromtimestamp(
            os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')

        # Extract hachoir metadata
        self._extract_hachoir_metadata(file_path)

        # Extract mediainfo metadata
        self._extract_mediainfo_metadata(file_path)

        # Extract ffmpeg metadata
        self._extract_ffmpeg_metadata(file_path)

        # Extract thumbnail and thumbnail metadata
        self._extract_thumbnail_metadata(file_path)

        # Extract creation history (new functionality)
        self._extract_creation_history(file_path)

        return self.metadata

    def extract_from_youtube(self, youtube_url):
        """Extract metadata from a YouTube video"""
        try:
            yt = YouTube(youtube_url)

            self.metadata['youtube'] = {
                'title': yt.title,
                'video_id': yt.video_id,
                'url': youtube_url,
                'author': yt.author,
                'channel_url': yt.channel_url,
                'channel_id': yt.channel_id,
                'publish_date': str(yt.publish_date) if yt.publish_date else None,
                'views': yt.views,
                'description': yt.description,
                'keywords': yt.keywords,
                'length': yt.length,
                'rating': yt.rating,
                'age_restricted': yt.age_restricted,
                'thumbnail_url': yt.thumbnail_url,
            }

            # Get available streams info
            streams_info = []
            for stream in yt.streams:
                stream_data = {
                    'itag': stream.itag,
                    'mime_type': stream.mime_type,
                    'resolution': stream.resolution,
                    'fps': stream.fps,
                    'bitrate': stream.bitrate,
                    'type': stream.type,
                    'codecs': stream.codecs,
                    'file_size': stream.filesize if hasattr(stream, 'filesize') else None,
                    'file_size_human': self._human_readable_size(stream.filesize) if hasattr(stream,
                                                                                             'filesize') else None,
                }
                streams_info.append(stream_data)

            self.metadata['youtube']['streams'] = streams_info

            # Download thumbnail for analysis
            if yt.thumbnail_url:
                thumbnail_path = os.path.join(self.temp_dir, f"{yt.video_id}_thumbnail.jpg")
                urllib.request.urlretrieve(yt.thumbnail_url, thumbnail_path)
                self._extract_image_exif(thumbnail_path, 'thumbnail_metadata')

            return self.metadata

        except Exception as e:
            self.metadata['error'] = f"Failed to extract YouTube metadata: {str(e)}"
            return self.metadata

    def _extract_hachoir_metadata(self, file_path):
        """Extract metadata using hachoir"""
        try:
            parser = createParser(file_path)
            if parser:
                metadata = extractMetadata(parser)
                if metadata:
                    data = {}
                    for line in metadata.exportPlaintext():
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            # Remove hachoir's indentation
                            key = key.lstrip('- ')
                            data[key] = value
                    self.metadata['hachoir'] = data
        except Exception as e:
            self.metadata['hachoir_error'] = str(e)

    def _extract_mediainfo_metadata(self, file_path):
        """Extract metadata using MediaInfo"""
        try:
            media_info = MediaInfo.parse(file_path)
            data = {'tracks': []}

            for track in media_info.tracks:
                track_data = {k: v for k, v in track.to_data().items()}
                data['tracks'].append(track_data)

            self.metadata['mediainfo'] = data
        except Exception as e:
            self.metadata['mediainfo_error'] = str(e)

    def _extract_ffmpeg_metadata(self, file_path):
        """Extract metadata using ffmpeg"""
        try:
            probe = ffmpeg.probe(file_path)
            self.metadata['ffmpeg'] = probe

            # Extract key video information for easy access
            if 'streams' in probe:
                video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
                audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']

                if video_streams:
                    main_video = video_streams[0]
                    video_info = {
                        'codec': main_video.get('codec_name'),
                        'width': main_video.get('width'),
                        'height': main_video.get('height'),
                        'bitrate': main_video.get('bit_rate'),
                        'fps': eval(main_video.get('avg_frame_rate', '0/1')) if '/' in main_video.get('avg_frame_rate',
                                                                                                      '0/1') else main_video.get(
                            'avg_frame_rate'),
                        'duration': main_video.get('duration'),
                    }
                    self.metadata['video_info'] = video_info

                if audio_streams:
                    main_audio = audio_streams[0]
                    audio_info = {
                        'codec': main_audio.get('codec_name'),
                        'channels': main_audio.get('channels'),
                        'sample_rate': main_audio.get('sample_rate'),
                        'bitrate': main_audio.get('bit_rate'),
                    }
                    self.metadata['audio_info'] = audio_info

        except Exception as e:
            self.metadata['ffmpeg_error'] = str(e)

    def _extract_thumbnail_metadata(self, file_path):
        """Extract thumbnail and its metadata"""
        try:
            thumbnail_path = os.path.join(self.temp_dir, f"thumbnail_{os.path.basename(file_path)}.jpg")

            # Extract thumbnail using ffmpeg
            ffmpeg.input(file_path, ss=1).filter('scale', 320, -1).output(
                thumbnail_path, vframes=1).overwrite_output().run(quiet=True)

            # Extract EXIF data from thumbnail
            if os.path.exists(thumbnail_path):
                self._extract_image_exif(thumbnail_path, 'thumbnail_metadata')
        except Exception as e:
            self.metadata['thumbnail_error'] = str(e)

    def _extract_image_exif(self, image_path, metadata_key):
        """Extract EXIF data from an image"""
        try:
            img = Image.open(image_path)
            exif_data = {}

            if hasattr(img, '_getexif') and img._getexif():
                for tag_id, value in img._getexif().items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            value = str(value)
                    exif_data[tag] = value

            # Add basic image info
            exif_data['format'] = img.format
            exif_data['mode'] = img.mode
            exif_data['size'] = img.size

            self.metadata[metadata_key] = exif_data
        except Exception as e:
            self.metadata[f"{metadata_key}_error"] = str(e)

    def _extract_creation_history(self, file_path):
        """Extract creation history and editing applications information"""
        creation_history = {
            'original_creation': None,
            'modifications': [],
            'applications_used': set()
        }

        # Get the original creation date from various metadata sources
        original_creation_date = None

        # Try to get from MediaInfo
        if 'mediainfo' in self.metadata:
            for track in self.metadata['mediainfo'].get('tracks', []):
                if track.get('encoded_date'):
                    date_str = track.get('encoded_date')
                    # Clean up the date format if needed
                    if date_str.startswith('UTC '):
                        date_str = date_str[4:]
                    try:
                        # Try to parse the date
                        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                        if original_creation_date is None or date_obj < original_creation_date:
                            original_creation_date = date_obj
                    except ValueError:
                        pass

        # Try to get from FFmpeg metadata
        if 'ffmpeg' in self.metadata:
            if 'format' in self.metadata['ffmpeg'] and 'tags' in self.metadata['ffmpeg']['format']:
                tags = self.metadata['ffmpeg']['format']['tags']
                creation_time = tags.get('creation_time')
                if creation_time:
                    try:
                        date_obj = datetime.datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                        if original_creation_date is None or date_obj < original_creation_date:
                            original_creation_date = date_obj
                    except ValueError:
                        pass

        # Try to get from hachoir
        if 'hachoir' in self.metadata:
            creation_date = self.metadata['hachoir'].get('Creation date')
            if creation_date:
                try:
                    # Format might vary, try common formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            date_obj = datetime.datetime.strptime(creation_date, fmt)
                            if original_creation_date is None or date_obj < original_creation_date:
                                original_creation_date = date_obj
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

        # Use file creation time as fallback
        if original_creation_date is None and 'creation_time' in self.metadata:
            try:
                original_creation_date = datetime.datetime.strptime(
                    self.metadata['creation_time'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass

        # Set the original creation date
        if original_creation_date:
            creation_history['original_creation'] = original_creation_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            creation_history['original_creation'] = "Unknown"

        # Find modification dates
        modification_dates = []

        # Check for modification date in MediaInfo
        if 'mediainfo' in self.metadata:
            for track in self.metadata['mediainfo'].get('tracks', []):
                if track.get('tagged_date') and track.get('tagged_date') != track.get('encoded_date'):
                    date_str = track.get('tagged_date')
                    if date_str.startswith('UTC '):
                        date_str = date_str[4:]
                    try:
                        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                        modification_dates.append({
                            'date': date_obj.strftime('%Y-%m-%d %H:%M:%S'),
                            'source': 'MediaInfo tagged_date'
                        })
                    except ValueError:
                        pass

        # Check for modification date in FFmpeg metadata
        if 'ffmpeg' in self.metadata:
            if 'format' in self.metadata['ffmpeg'] and 'tags' in self.metadata['ffmpeg']['format']:
                tags = self.metadata['ffmpeg']['format']['tags']
                if tags.get('modification_time') and tags.get('modification_time') != tags.get('creation_time'):
                    try:
                        date_obj = datetime.datetime.fromisoformat(tags.get('modification_time').replace('Z', '+00:00'))
                        modification_dates.append({
                            'date': date_obj.strftime('%Y-%m-%d %H:%M:%S'),
                            'source': 'FFmpeg modification_time'
                        })
                    except ValueError:
                        pass

        # Use file modification time
        if 'last_modified' in self.metadata and self.metadata['last_modified'] != self.metadata.get('creation_time'):
            modification_dates.append({
                'date': self.metadata['last_modified'],
                'source': 'File system'
            })

        # Sort modifications by date
        modification_dates.sort(key=lambda x: x['date'])
        creation_history['modifications'] = modification_dates

        # Extract software information
        software_info = set()

        # From FFmpeg
        if 'ffmpeg' in self.metadata:
            if 'format' in self.metadata['ffmpeg'] and 'tags' in self.metadata['ffmpeg']['format']:
                tags = self.metadata['ffmpeg']['format']['tags']
                if tags.get('encoder'):
                    software_info.add(tags.get('encoder'))
                if tags.get('handler_name'):
                    software_info.add(tags.get('handler_name'))

        # From MediaInfo
        if 'mediainfo' in self.metadata:
            for track in self.metadata['mediainfo'].get('tracks', []):
                if track.get('writing_application'):
                    software_info.add(track.get('writing_application'))
                if track.get('writing_library'):
                    software_info.add(track.get('writing_library'))
                if track.get('encoded_library_name'):
                    software_info.add(track.get('encoded_library_name'))

        # From hachoir
        if 'hachoir' in self.metadata:
            if self.metadata['hachoir'].get('Producer'):
                software_info.add(self.metadata['hachoir'].get('Producer'))
            if self.metadata['hachoir'].get('Software'):
                software_info.add(self.metadata['hachoir'].get('Software'))
            if self.metadata['hachoir'].get('Encoder'):
                software_info.add(self.metadata['hachoir'].get('Encoder'))

        creation_history['applications_used'] = list(software_info)

        # Add creation history to metadata
        self.metadata['creation_history'] = creation_history

    def _human_readable_size(self, size_bytes):
        """Convert bytes to human readable format"""
        if size_bytes is None:
            return None

        size_bytes = int(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024 or unit == 'TB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024

    def save_metadata_to_file(self, output_file=None):
        """Save metadata to a JSON file"""
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if 'filename' in self.metadata:
                base_name = os.path.splitext(self.metadata['filename'])[0]
                output_file = f"{base_name}_metadata_{timestamp}.json"
            elif 'youtube' in self.metadata and 'video_id' in self.metadata['youtube']:
                output_file = f"youtube_{self.metadata['youtube']['video_id']}_metadata_{timestamp}.json"
            else:
                output_file = f"video_metadata_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        return output_file

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")


class VideoMetadataExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Metadata Extractor")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create tabs
        self.tab_local = ttk.Frame(self.notebook)
        self.tab_youtube = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_local, text="Local Video")
        self.notebook.add(self.tab_youtube, text="YouTube Video")

        # Setup Local Video tab
        self.setup_local_tab()

        # Setup YouTube tab
        self.setup_youtube_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create extractor instance
        self.extractor = VideoMetadataExtractor()

    def setup_local_tab(self):
        # Frame for file selection
        file_frame = ttk.Frame(self.tab_local, padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # File path label
        ttk.Label(file_frame, text="Video File:").pack(side=tk.LEFT)

        # File path entry
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Browse button
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.LEFT, padx=5)

        # Extract button
        extract_button = ttk.Button(file_frame, text="Extract Metadata", command=self.extract_local)
        extract_button.pack(side=tk.LEFT, padx=5)

        # Results frame with tabs for different views
        results_frame = ttk.LabelFrame(self.tab_local, text="Metadata Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create notebook for results tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Full metadata tab
        full_metadata_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(full_metadata_tab, text="Full Metadata")

        self.local_results = scrolledtext.ScrolledText(full_metadata_tab, wrap=tk.WORD)
        self.local_results.pack(fill=tk.BOTH, expand=True)

        # Creation history tab
        creation_history_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(creation_history_tab, text="Creation History")

        self.creation_history_results = scrolledtext.ScrolledText(creation_history_tab, wrap=tk.WORD)
        self.creation_history_results.pack(fill=tk.BOTH, expand=True)

        # Save button
        save_frame = ttk.Frame(self.tab_local, padding="10")
        save_frame.pack(fill=tk.X)

        save_button = ttk.Button(save_frame, text="Save Results", command=lambda: self.save_results('local'))
        save_button.pack(side=tk.RIGHT)

    def setup_youtube_tab(self):
        # Frame for URL entry
        url_frame = ttk.Frame(self.tab_youtube, padding="10")
        url_frame.pack(fill=tk.X, pady=5)

        # URL label
        ttk.Label(url_frame, text="YouTube URL:").pack(side=tk.LEFT)

        # URL entry
        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=50)
        url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Extract button
        extract_button = ttk.Button(url_frame, text="Extract Metadata", command=self.extract_youtube)
        extract_button.pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.tab_youtube, text="Metadata Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Text widget for results
        self.youtube_results = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.youtube_results.pack(fill=tk.BOTH, expand=True)

        # Save button
        save_frame = ttk.Frame(self.tab_youtube, padding="10")
        save_frame.pack(fill=tk.X)

        save_button = ttk.Button(save_frame, text="Save Results", command=lambda: self.save_results('youtube'))
        save_button.pack(side=tk.RIGHT)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                ("All files", "*.*")
            )
        )
        if file_path:
            self.file_path_var.set(file_path)

    def extract_local(self):
        file_path = self.file_path_var.get()
        if not file_path:
            self.show_message("Please select a video file")
            return

        if not os.path.exists(file_path):
            self.show_message("File does not exist")
            return

        self.status_var.set("Extracting metadata from local file...")
        self.root.update_idletasks()

        try:
            # Clear previous results
            self.local_results.delete('1.0', tk.END)
            self.creation_history_results.delete('1.0', tk.END)

            metadata = self.extractor.extract_from_local_file(file_path)
            formatted_data = json.dumps(metadata, indent=2, default=str)

            self.local_results.insert(tk.END, formatted_data)

            # Format creation history for the creation history tab
            if 'creation_history' in metadata:
                history = metadata['creation_history']

                creation_history_text = "=== VIDEO CREATION HISTORY ===\n\n"

                # Original creation date
                creation_history_text += f"ORIGINAL CREATION: {history['original_creation']}\n\n"

                # Applications used
                creation_history_text += "APPLICATIONS USED:\n"
                if history['applications_used']:
                    for app in history['applications_used']:
                        creation_history_text += f"- {app}\n"
                else:
                    creation_history_text += "- No software information detected\n"

                # Online tools used (new section)
                creation_history_text += "\nONLINE TOOLS DETECTED:\n"
                if 'online_tools' in history and history['online_tools']:
                    for tool in history['online_tools']:
                        creation_history_text += f"- {tool}\n"
                else:
                    creation_history_text += "- No online tools detected\n"

                # Modifications
                creation_history_text += "\nMODIFICATIONS:\n"
                if history['modifications']:
                    for i, mod in enumerate(history['modifications'], 1):
                        creation_history_text += f"{i}. Date: {mod['date']}\n   Source: {mod['source']}\n"
                else:
                    creation_history_text += "- No modifications detected\n"

                self.creation_history_results.insert(tk.END, creation_history_text)

            self.status_var.set(f"Metadata extracted successfully from {os.path.basename(file_path)}")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.local_results.insert(tk.END, f"Error extracting metadata: {str(e)}")

    def extract_youtube(self):
        url = self.url_var.get()
        if not url or not url.startswith(("http://", "https://")):
            self.show_message("Please enter a valid YouTube URL")
            return

        self.status_var.set("Extracting metadata from YouTube video...")
        self.root.update_idletasks()

        try:
            # Clear previous results
            self.youtube_results.delete('1.0', tk.END)

            metadata = self.extractor.extract_from_youtube(url)
            formatted_data = json.dumps(metadata, indent=2, default=str)

            self.youtube_results.insert(tk.END, formatted_data)

            if 'error' in metadata:
                self.status_var.set(f"Error: {metadata['error']}")
            else:
                video_id = metadata.get('youtube', {}).get('video_id', 'unknown')
                self.status_var.set(f"Metadata extracted successfully from YouTube video: {video_id}")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.youtube_results.insert(tk.END, f"Error extracting metadata: {str(e)}")

    def save_results(self, tab_name):
        if tab_name == 'local':
            current_tab_index = self.results_notebook.index(self.results_notebook.select())
            if current_tab_index == 0 and not self.local_results.get('1.0', tk.END).strip():
                self.show_message("No data to save")
                return
            elif current_tab_index == 1 and not self.creation_history_results.get('1.0', tk.END).strip():
                self.show_message("No data to save")
                return

        if tab_name == 'youtube' and not self.youtube_results.get('1.0', tk.END).strip():
            self.show_message("No data to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Metadata",
            defaultextension=".json" if tab_name != 'local' or self.results_notebook.index(
                self.results_notebook.select()) != 1 else ".txt",
            filetypes=(("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*"))
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if tab_name == 'local':
                    current_tab_index = self.results_notebook.index(self.results_notebook.select())
                    if current_tab_index == 0:  # Full metadata tab
                        f.write(self.local_results.get('1.0', tk.END))
                    else:  # Creation history tab
                        f.write(self.creation_history_results.get('1.0', tk.END))
                else:
                    f.write(self.youtube_results.get('1.0', tk.END))

            self.status_var.set(f"Metadata saved to {file_path}")
        except Exception as e:
            self.status_var.set(f"Error saving file: {str(e)}")

    def show_message(self, message):
        self.status_var.set(message)


def main():
    root = tk.Tk()
    app = VideoMetadataExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
#
# from pymediainfo import MediaInfo
#
#
# def get_video_software_info(file_path):
#     media_info = MediaInfo.parse(file_path)
#     found = False
#
#     for track in media_info.tracks:
#         if track.track_type == "General":
#             print("📁 File:", file_path)
#             print("🧾 Format:", track.format)
#
#             # These are the most common fields for software info
#             if track.encoded_by:
#                 print("🛠️ Encoded by:", track.encoded_by)
#                 found = True
#             if track.writing_application:
#                 print("🛠️ Writing application:", track.writing_application)
#                 found = True
#             if track.writing_library:
#                 print("🛠️ Writing library:", track.writing_library)
#                 found = True
#
#             # Print all available metadata if nothing found
#             if not found:
#                 print("⚠️ No explicit software info found. Dumping all available general metadata:")
#                 for key, value in track.to_data().items():
#                     if isinstance(value, str) and any(
#                             kw in key.lower() for kw in ["application", "encoder", "software", "library", "tool"]):
#                         print(f"{key}: {value}")
#             break
#
#
# # Example usage
# get_video_software_info(r"D:\enjoy2.mp4")
# Lavf57.83.100 means someone use ffmpeg and CLI to develop that videob means fake
