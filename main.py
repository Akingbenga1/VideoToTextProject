import gc
import logging

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, List, Dict, Any

import openai
import yaml
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

import json

from langchain_core.messages import ToolMessage

import whisper

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']



log_level = "DEBUG"
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')



class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


llm = ChatOpenAI(model="gpt-4")

tool = TavilySearchResults(max_results=2)
tools = [tool]
# tool.invoke("What's a 'node' in LangGraph?")

llm_with_tools = llm.bind_tools(tools)

graph_builder.add_node("chatbot", chatbot)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("value:", value)
            print("Assistant:", value["messages"][-1].content)

def process_main(video_path, diarize = True ):
    global transcription_text
    if not video_path:
        print("No file input");
        return []

    start_time = time.monotonic()
    paths = [video_path]
    results = []

    for path in paths:
        try:
            if diarize:
                audio_file, segments = perform_transcription(video_path, 2, llm_with_tools, 3,
                                                             diarize=True)
                transcription_text = {'audio_file': audio_file, 'transcription': segments}
        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            continue
    logging.debug("Total time taken: %s seconds", time.monotonic() - start_time)
    logging.info("MAIN: returing transcription_text.")
    return transcription_text

def perform_transcription(video_path, offset = 3, whisper_model = "llm", vad_filter = 2, diarize=False, combined_format = False):
    temp_files = []
    logging.info(f"Processing media: {video_path}")
    global segments_json_path
    audio_file_path = convert_to_wav(video_path, offset)
    logging.debug(f"Converted audio file: {audio_file_path}")
    temp_files.append(audio_file_path)
    logging.debug("Replacing audio file with segments.json file")
    segments_json_path = audio_file_path.replace('.wav', '.segments.json')
    temp_files.append(segments_json_path)

    if diarize:
        diarized_json_path = audio_file_path.replace('.wav', '.diarized.json')

        # Check if diarized JSON already exists
        if os.path.exists(diarized_json_path):
            logging.info(f"Diarized file already exists: {diarized_json_path}")
            try:
                with open(diarized_json_path, 'r') as file:
                    diarized_segments = json.load(file)
                if not diarized_segments:
                    logging.warning(f"Diarized JSON file is empty, re-generating: {diarized_json_path}")
                    raise ValueError("Empty diarized JSON file")
                logging.debug(f"Loaded diarized segments from {diarized_json_path}")
                return audio_file_path, diarized_segments
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Failed to read or parse the diarized JSON file: {e}")
                os.remove(diarized_json_path)

        # If diarized file doesn't exist or was corrupted, generate new diarized transcription
        logging.info(f"Generating diarized transcription for {audio_file_path}")
        diarized_segments = combine_transcription_and_diarization(audio_file_path)


        # Save diarized segments
        with open(diarized_json_path, 'w') as file:
            json.dump(diarized_segments, file, indent=2)

        return audio_file_path, diarized_segments

    # Non-diarized transcription (existing functionality)
    if os.path.exists(segments_json_path):
        logging.info(f"Segments file already exists: {segments_json_path}")
        try:
            with open(segments_json_path, 'r') as file:
                print("We got hwre too ooo ")
                segments = json.load(file)
            if not segments:
                logging.warning(f"Segments JSON file is empty, re-generating: {segments_json_path}")
                raise ValueError("Empty segments JSON file")
            logging.debug(f"Loaded segments from {segments_json_path}")
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to read or parse the segments JSON file: {e}")
            os.remove(segments_json_path)
            logging.info(f"Re-generating transcription for {audio_file_path}")
            # audio_file, segments = re_generate_transcription(audio_file_path, whisper_model, vad_filter)
            if segments is None:
                return None, None
    else:
        audio_file, segments = re_generate_transcription(audio_file_path, whisper_model, vad_filter)

    return audio_file_path, segments

def load_pipeline_from_pretrained(path_to_config: str | Path) -> SpeakerDiarization:
    path_to_config = Path(path_to_config).resolve()
    logging.debug(f"Loading pyannote pipeline from {path_to_config}...")

    if not path_to_config.exists():
        raise FileNotFoundError(f"Config file not found: {path_to_config}")

    # Load the YAML configuration
    with open(path_to_config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Debug: print the entire config
    logging.debug(f"Loaded config: {config}")

    # Create the SpeakerDiarization pipeline
    try:
        pipeline = SpeakerDiarization(
            segmentation=config['pipeline']['params']['segmentation'],
            embedding=config['pipeline']['params']['embedding'],
            # embedding_batch_size=config['pipeline']['params']['embedding_batch_size'],
            clustering=config['pipeline']['params']['clustering'],
        )
    except KeyError as e:
        logging.error(f"Error accessing config key: {e}")
        raise

    # Set other parameters
    try:
        pipeline_params = {
            "segmentation": {},
            "clustering": {},
        }

        if 'params' in config and 'segmentation' in config['params']:
            if 'min_duration_off' in config['params']['segmentation']:
                pipeline_params["segmentation"]["min_duration_off"] = config['params']['segmentation']['min_duration_off']

        if 'params' in config and 'clustering' in config['params']:
            if 'method' in config['params']['clustering']:
                pipeline_params["clustering"]["method"] = config['params']['clustering']['method']
            if 'min_cluster_size' in config['params']['clustering']:
                pipeline_params["clustering"]["min_cluster_size"] = config['params']['clustering']['min_cluster_size']
            if 'threshold' in config['params']['clustering']:
                pipeline_params["clustering"]["threshold"] = config['params']['clustering']['threshold']

        if 'pipeline' in config and 'params' in config['pipeline']:
            if 'embedding_batch_size' in config['pipeline']['params']:
                pass
                # pipeline_params["embedding_batch_size"] = config['pipeline']['params']['embedding_batch_size']
            if 'embedding_exclude_overlap' in config['pipeline']['params']:
                pass
                # pipeline_params["embedding_exclude_overlap"] = config['pipeline']['params']['embedding_exclude_overlap']
            if 'segmentation_batch_size' in config['pipeline']['params']:
                pass
                # pipeline_params["segmentation_batch_size"] = config['pipeline']['params']['segmentation_batch_size']

        logging.debug(f"Pipeline params: {pipeline_params}")
        pipeline.instantiate(pipeline_params)
    except KeyError as e:
        logging.error(f"Error accessing config key: {e}")
        raise
    except Exception as e:
        logging.error(f"Error instantiating pipeline: {e}")
        raise

    return pipeline

def audio_diarization(audio_file_path: str) -> list:
    logging.info('audio-diarization: Loading pyannote pipeline')

    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / 'models' / 'pyannote_diarization_config.yaml'
    logging.info(f"audio-diarization: Loading pipeline from {config_path}")

    try:
        pipeline = load_pipeline_from_pretrained(config_path)
    except Exception as e:
        logging.error(f"Failed to load pipeline: {str(e)}")
        raise

    logging.info(f"audio-diarization: Audio file path: {audio_file_path}")

    try:
        logging.info('audio-diarization: Starting diarization...')
        diarization_result = pipeline(audio_file_path)
        logging.info(str(diarization_result))
        # print(str(diarization_result))

        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segment = {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            }
            logging.debug(f"Segment: {segment}")
            segments.append(segment)
        logging.info("audio-diarization: Diarization completed with pyannote")

        return segments

    except Exception as e:
        logging.error(f"audio-diarization: Error performing diarization: {str(e)}")
        raise RuntimeError("audio-diarization: Error performing diarization") from e

def re_generate_transcription(audio_file_path, whisper_model, vad_filter):
    try:
        segments = speech_to_text(audio_file_path, whisper_model=whisper_model, vad_filter=vad_filter)
        # Save segments to JSON
        with open(segments_json_path, 'w') as file:
            json.dump(segments, file, indent=2)
        logging.debug(f"Transcription segments saved to {segments_json_path}")
        return audio_file_path, segments
    except Exception as e:
        logging.error(f"Error in re-generating transcription: {str(e)}")
        return None, None

def speech_to_text(audio_file_path, selected_source_lang='en', whisper_model = "turbo", vad_filter=False, diarize=False):
    global  processing_choice
    logging.info('speech-to-text: Loading faster_whisper model: %s', whisper_model)

    time_start = time.time()
    if audio_file_path is None:
        raise ValueError("speech-to-text: No audio file provided")
    logging.info("speech-to-text: Audio file path: %s", audio_file_path)

    try:
        _, file_ending = os.path.splitext(audio_file_path)
        out_file = audio_file_path.replace(file_ending, "-whisper_model-"+whisper_model+".segments.json")
        prettified_out_file = audio_file_path.replace(file_ending, "-whisper_model-"+whisper_model+".segments_pretty.json")
        if os.path.exists(out_file):
            logging.info("speech-to-text: Segments file already exists: %s", out_file)
            with open(out_file) as f:
                global segments
                segments = json.load(f)
            return segments

        logging.info('speech-to-text: Starting transcription...')
        options = dict(language=selected_source_lang, beam_size=5, best_of=5, vad_filter=vad_filter)
        transcribe_options = dict(task="transcribe", **options)
        # use function and config at top of file
        logging.debug("speech-to-text: Using whisper model: %s", whisper_model)
        model = whisper.load_model("turbo")
        # whisper_model_instance = get_whisper_model_model(whisper_model, processing_choice)
        segments_raw = model.transcribe(audio_file_path)#, **transcribe_options)
        logging.info("speech-to-text: Starting transcription: %s %s", str(segments_raw), type(segments_raw))
        # logging.info(str(info))

        segments = []
        for segment_chunk in segments_raw["segments"]:
            chunk = {
                "Time_Start": segment_chunk["start"],
                "Time_End": segment_chunk["end"],
                "Text": segment_chunk["text"]
            }
            logging.debug("Segment: %s", chunk)
            segments.append(chunk)
            # Print to verify its working
            print(f"{segment_chunk['start']:.2f}s - {segment_chunk['end']:.2f}s | {segment_chunk['text']}")

            # Log it as well.
            logging.debug(
                f"Transcribed Segment: {segment_chunk['start']:.2f}s - {segment_chunk['end']:.2f}s | {segment_chunk['text']}")

        if segments:
            zeroth_chunk = {
                                "Time_Start": -0.0,
                                "Time_End": -0.0,
                                "Text": f"This text was transcribed using openai "
                            }
            # segments[0]["Text"] = f"This text was transcribed using openai "
            segments.insert(0, zeroth_chunk)

        if not segments:
            raise RuntimeError("No transcription produced. The audio file may be invalid or empty.")
        logging.info("speech-to-text: Transcription completed in %.2f seconds", time.time() - time_start)

        # Save the segments to a JSON file - prettified and non-prettified
        # FIXME so this is an optional flag to save either the prettified json file or the normal one
        save_json = True
        if save_json:
            logging.info("speech-to-text: Saving segments to JSON file")
            output_data = {'segments': segments}

            logging.info("speech-to-text: Saving prettified JSON to %s", prettified_out_file)
            with open(prettified_out_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            logging.info("speech-to-text: Saving JSON to %s", out_file)
            with open(out_file, 'w') as f:
                json.dump(output_data, f)

        logging.debug(f"speech-to-text: returning {segments[:500]}")
        gc.collect()
        return segments

    except Exception as e:
        logging.error("speech-to-text: Error transcribing audio: %s", str(e))
        raise RuntimeError("speech-to-text: Error transcribing audio")

# Old
# def audio_diarization(audio_file_path):
#     logging.info('audio-diarization: Loading pyannote pipeline')
#
#     #config file loading
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # Construct the path to the config file
#     config_path = os.path.join(current_dir, 'Config_Files', 'config.txt')
#     # Read the config file
#     config = configparser.ConfigParser()
#     config.read(config_path)
#     processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')
#
#     base_dir = Path(__file__).parent.resolve()
#     config_path = base_dir / 'models' / 'config.yaml'
#     pipeline = load_pipeline_from_pretrained(config_path)
#
#     time_start = time.time()
#     if audio_file_path is None:
#         raise ValueError("audio-diarization: No audio file provided")
#     logging.info("audio-diarization: Audio file path: %s", audio_file_path)
#
#     try:
#         _, file_ending = os.path.splitext(audio_file_path)
#         out_file = audio_file_path.replace(file_ending, ".diarization.json")
#         prettified_out_file = audio_file_path.replace(file_ending, ".diarization_pretty.json")
#         if os.path.exists(out_file):
#             logging.info("audio-diarization: Diarization file already exists: %s", out_file)
#             with open(out_file) as f:
#                 global diarization_result
#                 diarization_result = json.load(f)
#             return diarization_result
#
#         logging.info('audio-diarization: Starting diarization...')
#         diarization_result = pipeline(audio_file_path)
#
#         segments = []
#         for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#             chunk = {
#                 "Time_Start": turn.start,
#                 "Time_End": turn.end,
#                 "Speaker": speaker
#             }
#             logging.debug("Segment: %s", chunk)
#             segments.append(chunk)
#         logging.info("audio-diarization: Diarization completed with pyannote")
#
#         output_data = {'segments': segments}
#
#         logging.info("audio-diarization: Saving prettified JSON to %s", prettified_out_file)
#         with open(prettified_out_file, 'w') as f:
#             json.dump(output_data, f, indent=2)
#
#         logging.info("audio-diarization: Saving JSON to %s", out_file)
#         with open(out_file, 'w') as f:
#             json.dump(output_data, f)
#
#     except Exception as e:
#         logging.error("audio-diarization: Error performing diarization: %s", str(e))
#         raise RuntimeError("audio-diarization: Error performing diarization")
#     return segments
def combine_transcription_and_diarization(audio_file_path: str) -> List[Dict[str, Any]]:
    logging.info('combine-transcription-and-diarization: Starting transcription and diarization...')

    try:
        logging.info('Performing speech-to-text...')
        logging.info(f"Transcription result type: {str(audio_file_path)}")
        transcription_result = speech_to_text(audio_file_path)
        logging.info(f"Transcription result type: {type(transcription_result)}")
        logging.info(f"Transcription result: {transcription_result[:3] if isinstance(transcription_result, list) and len(transcription_result) > 3 else transcription_result}")

        logging.info('Performing audio diarization...')
        # diarization_result = audio_diarization(audio_file_path)
        diarization_result = []
        logging.info(f"Diarization result type: {type(diarization_result)}")
        logging.info(f"Diarization result sample: {diarization_result[:3] if isinstance(diarization_result, list) and len(diarization_result) > 3 else diarization_result}")

        if not transcription_result:
            logging.error("Empty result from transcription")
            return []

        if not diarization_result:
            logging.error("Empty result from diarization")
            # return []

        # Handle the case where transcription_result is a dict with a 'segments' key
        if isinstance(transcription_result, dict) and 'segments' in transcription_result:
            transcription_segments = transcription_result['segments']
        elif isinstance(transcription_result, list):
            transcription_segments = transcription_result
        else:
            logging.error(f"Unexpected transcription result format: {type(transcription_result)}")
            return []

        logging.info(f"Number of transcription segments: {len(transcription_segments)}")
        logging.info(f"Transcription segments sample: {transcription_segments[:3] if len(transcription_segments) > 3 else transcription_segments}")

        if not isinstance(diarization_result, list):
            logging.error(f"Unexpected diarization result format: {type(diarization_result)}")
            # return []
        # print(transcription_segments)
        # print(diarization_result)
        combined_result = []
        for transcription_segment in transcription_segments:
            if not isinstance(transcription_segment, dict):
                logging.warning(f"Unexpected transcription segment format: {transcription_segment}")
                continue

            for diarization_segment in diarization_result:
                if not isinstance(diarization_segment, dict):
                    logging.warning(f"Unexpected diarization segment format: {diarization_segment}")
                    continue

                try:
                    trans_start = round ( transcription_segment.get('Time_Start', 0) )
                    trans_end = round ( transcription_segment.get('Time_End', 0) )
                    diar_start = round (diarization_segment.get('start', 0) )
                    diar_end = round(diarization_segment.get('end', 0))

                    if trans_start >= diar_start and trans_end <= diar_end:
                        combined_segment = {
                            "Time_Start": trans_start,
                            "Time_End": trans_end,
                            "Speaker": diarization_segment.get('speaker', 'Unknown'),
                            "Text": transcription_segment.get('Text', '')
                        }
                        combined_result.append(combined_segment)
                        break
                except Exception as e:
                    logging.error(f"Error processing segment: {str(e)}")
                    logging.error(f"Transcription segment: {transcription_segment}")
                    logging.error(f"Diarization segment: {diarization_segment}")
                    continue

        logging.info(f"Combined result length: {len(combined_result)}")
        logging.info(f"Combined result sample: {combined_result[:3] if len(combined_result) > 3 else combined_result}")
        return transcription_segments

    except Exception as e:
        logging.error(f"Error in combine_transcription_and_diarization: {str(e)}", exc_info=True)
        return []

def convert_to_wav(video_file_path, offset=0, overwrite=True):
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    if os.path.exists(out_path) and not overwrite:
        print(f"File '{out_path}' already exists. Skipping conversion.")
        logging.info(f"Skipping conversion as file already exists: {out_path}")
        return out_path
    print("Starting conversion process of .m4a to .WAV")
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    try:
        if os.name == "nt":
            logging.debug("ffmpeg being ran on windows")

            if sys.platform.startswith('win'):
                ffmpeg_cmd = ".\\Bin\\ffmpeg.exe"
                logging.debug(f"ffmpeg_cmd: {ffmpeg_cmd}")
            else:
                ffmpeg_cmd = 'ffmpeg'  # Assume 'ffmpeg' is in PATH for non-Windows systems

            command = [
                ffmpeg_cmd,  # Assuming the working directory is correctly set where .\Bin exists
                "-ss", "00:00:00",  # Start at the beginning of the video
                "-i", video_file_path,
                "-ar", "16000",  # Audio sample rate
                "-ac", "1",  # Number of audio channels
                "-c:a", "pcm_s16le",  # Audio codec
                out_path
            ]
            try:
                # Redirect stdin from null device to prevent ffmpeg from waiting for input
                with open(os.devnull, 'rb') as null_file:
                    result = subprocess.run(command, stdin=null_file, text=True, capture_output=True)
                if result.returncode == 0:
                    logging.info("FFmpeg executed successfully")
                    logging.debug("FFmpeg output: %s", result.stdout)
                else:
                    logging.error("Error in running FFmpeg")
                    logging.error("FFmpeg stderr: %s", result.stderr)
                    raise RuntimeError(f"FFmpeg error: {result.stderr}")
            except Exception as e:
                logging.error("Error occurred - ffmpeg doesn't like windows")
                raise RuntimeError("ffmpeg failed")
        elif os.name == "posix":
            os.system(f'ffmpeg -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
        else:
            raise RuntimeError("Unsupported operating system")
        logging.info("Conversion to WAV completed: %s", out_path)
    except subprocess.CalledProcessError as e:
        logging.error("Error executing FFmpeg command: %s", str(e))
        raise RuntimeError("Error converting video file to WAV")
    except Exception as e:
        logging.error("speech-to-text: Error transcribing audio: %s", str(e))
        return {"error": str(e)}
    gc.collect()
    return out_path


final_transcribed_result  = perform_transcription("spike_1_99_Mobile.mp4", 2 , llm_with_tools , 3 ,True, True )
# print(final_transcribed_result[1])
#
if True:
    # convert segments to a flat text and return
    print("I will combine the text here")
    segments_combined_text = " ".join([d['Text'] for d in final_transcribed_result[1]])
    print( segments_combined_text )


while False:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break