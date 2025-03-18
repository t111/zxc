#!/usr/bin/env python3
import argparse
import asyncio
import importlib.util
import json
import os
import sys
import traceback
from pprint import pprint

from langchain.schema import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style

# Constants for text styling
STYLE = Style.from_dict(
    {
        "human": "#00ff00 bold",
        "thinking": "#ff00ff italic",
        # "code": "#ffffff",
        "bot": "bold",
        "info": "#888888 italic",
        "multi": "#ff8800 bold",
        "system": "#888888",
    }
)


def load_graph_from_path(path_spec: str) -> CompiledStateGraph:
    """
    Dynamically load a graph from a path specification of the form:
      'path/to/file.py:variable_name'
    """
    file_path, variable = path_spec.split(":")

    spec = importlib.util.spec_from_file_location("module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["module"] = module
    spec.loader.exec_module(module)

    return getattr(module, variable)


def get_graph(graph_name: str) -> CompiledStateGraph:
    """
    Return the graph based on the provided name from the langgraph.json configuration.
    """
    config_path = os.path.abspath("langgraph.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON in configuration file {config_path}: {err}")

    graphs = config.get("graphs", {})
    if graph_name not in graphs:
        valid_names = ", ".join(f"'{name}'" for name in graphs.keys())
        raise ValueError(
            f"Invalid graph name: '{graph_name}'. Valid options are: {valid_names}"
        )

    return load_graph_from_path(graphs[graph_name])


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Setup and return the argument parser based on the available graph names.
    """
    try:
        with open("./langgraph.json", "r") as f:
            config = json.load(f)
        graph_names = list(config.get("graphs", {}).keys())
    except (FileNotFoundError, json.JSONDecodeError):
        graph_names = []

    parser = argparse.ArgumentParser(description="Chat interface for different graphs")
    parser.add_argument(
        "-g",
        "--graph_name",
        choices=graph_names,
        default=graph_names[0] if graph_names else None,
        help=f'The name of the graph to use ({", ".join(graph_names)})',
    )
    return parser


def print_formatted(message: str, style: Style, end: str = "\n") -> None:
    """Print the message with formatting."""
    print_formatted_text(HTML(message), style=style, end=end)


def find_project_root(start: str) -> str:
    """
    Traverse upward from the 'start' directory until a pyproject.toml is found.
    """
    current = os.path.abspath(start)
    while not os.path.isfile(os.path.join(current, "pyproject.toml")):
        parent = os.path.dirname(current)
        if parent == current:  # Reached filesystem root
            raise FileNotFoundError(
                "pyproject.toml not found in any ancestor directories"
            )
        current = parent
    return current


async def async_chat(graph: CompiledStateGraph) -> None:
    """
    Chat loop using prompt-toolkit for async input/output.
    """
    session = PromptSession()

    print_formatted(f"<info>Using graph: {graph.name}</info>", STYLE)
    print_formatted(
        "<info>Welcome! Start with ` to enter multi-line mode.</info>", STYLE
    )
    print_formatted(
        "<info>In multi-line mode, type a line with just ` to end input.</info>", STYLE
    )
    print_formatted(
        '<info>Type "exit", "quit", or "bye" to end the chat.</info>\n', STYLE
    )

    state: MessagesState = MessagesState(messages=[])

    async def get_input() -> str:
        """
        Return user input. Multi-line mode is enabled if the first line starts with `.
        """
        lines = []
        first_line = await session.prompt_async(
            HTML("<human>You: </human>"), style=STYLE
        )
        first_line = first_line.strip()
        if first_line.startswith("`"):
            lines.append(first_line[1:])
            while True:
                line = await session.prompt_async(
                    HTML("<multi>.... </multi>"), style=STYLE
                )
                line = line.strip()
                if line == "`":
                    break
                lines.append(line)
            return "\n".join(lines)
        return first_line

    try:
        while True:
            user_input = await get_input()
            # user_input = "NYC weather today?"
            if user_input.lower() in ["exit", "quit", "bye"]:
                sys.exit(0)

            if user_input.lower() == '/history':
                for msg in state['messages']:
                    msg.pretty_print()
                continue

            if user_input:
                print("")
                state["messages"].append(HumanMessage(content=user_input))

                # word_count = sum(len(msg.content.split()) for msg in state["messages"])
                # print_formatted(
                #     f'<info>Context size: {len(state["messages"])} messages, {word_count} words</info>',
                #     STYLE,
                # )
                print_formatted("<info>Invoking Agent...</info>\n", STYLE)

                state = await stream_output(graph, state)
                print("\n")

    except KeyboardInterrupt:
        sys.exit(15)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print_formatted(
            f"<bot>Bot: An error occurred, but we will continue: {e}</bot>", STYLE
        )


# async def stream_output_2(graph: CompiledStateGraph, state) -> MessagesState:
#     async for event in graph.astream_events(state, version="v2"):


async def stream_output(graph: CompiledStateGraph, state) -> MessagesState:
    stream_buffer = ''
    mode = 'bot'
    # i = 0
    async for t, data in graph.astream(state, stream_mode=["values", "messages"]):
        if t == 'values':
            state = data
        elif t == "messages":
            for chunk in data:
                if isinstance(chunk, AIMessageChunk):
                    messages = chunk.content
                    for msg in messages:
                        text = msg.get("text", "")
                        stream_buffer += text
                    while '\n' in stream_buffer:
                        line, stream_buffer = stream_buffer.split("\n", maxsplit=1)
                        if line.startswith("```thinking"):
                            mode = "thinking"
                        elif line.startswith("```"):
                            if mode == "bot":
                                mode = "code"
                            else:
                                mode = "bot"

                        print_formatted(f"<{mode}>{line}</{mode}>", STYLE, )  # print(event.keys())

                elif isinstance(chunk, ToolMessage):
                    print('\n')
                    chunk.pretty_print()
                    print('\n')
                elif not isinstance(chunk, dict):
                    print(type(chunk), chunk)
    print_formatted(f"<bot>{stream_buffer}</bot>", STYLE, '')  # print(event.keys())
    return state


def main() -> None:
    """
    Main entry point. Determines project root, loads configuration, and runs the chat.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = find_project_root(script_dir)
        print(f"project_root={project_root}")
        os.chdir(project_root)

        parser = setup_argument_parser()
        args = parser.parse_args()

        if not args.graph_name:
            raise SystemExit("No graph defined in configuration. Exiting.")

        selected_graph = get_graph(args.graph_name)
        asyncio.run(async_chat(selected_graph))
    except Exception as exc:
        print(traceback.format_exc())
        # print(f"Error: {exc}")
        sys.exit(1)
    finally:
        print_formatted("\n<bot>Bot: Goodbye!</bot>", STYLE)


if __name__ == "__main__":
    main()
