#!/usr/bin/env python3
"""
Main entry point for the LLM RL Framework

This script runs the framework with the specified configuration file.
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import traceback

from src.core.config import FrameworkConfig, LLMConfig, EpisodeControlConfig
from src.core.rollout_manager import RolloutManager
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(), show_time=True)]
    )


def load_config(config_file: str) -> FrameworkConfig:
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        return FrameworkConfig(**config_data)
    
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def display_config_summary(config: FrameworkConfig, console: Console) -> None:
    """Display configuration summary"""
    table = Table(title="Framework Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Environments", str(len(config.environments)))
    table.add_row("LLM Model", config.llm_config.model if config.llm_config else "Not configured")
    table.add_row("Max Episodes", str(config.episode_control_config.max_episodes))
    table.add_row("Stop on Success", str(config.episode_control_config.stop_on_success))
    table.add_row("Max Parallel Rollouts", str(config.rollout_config.max_parallel_rollouts))
    table.add_row("Global Timeout", f"{config.timeout_config.global_timeout}s")
    table.add_row("Output Path", config.rollout_config.trajectory_output_path)
    table.add_row("Plugins Enabled", str(config.rollout_config.enable_plugins))
    
    console.print(table)
    console.print()
    
    # Display environment details
    for env in config.environments:
        env_panel = Panel(
            f"[bold]ID:[/bold] {env.id}\n"
            f"[bold]Docker Image:[/bold] {env.docker_image}\n"
            f"[bold]Working Dir:[/bold] {env.working_directory}\n"
            f"[bold]Init Commands:[/bold] {len(env.init_commands)}\n"
            f"[bold]Unit Tests:[/bold] {len(env.unit_tests)}\n"
            f"[bold]Prompt Length:[/bold] {len(env.prompt)} chars",
            title=f"Environment: {env.id}",
            border_style="blue"
        )
        console.print(env_panel)


def display_results_summary(results: list, rollout_manager: RolloutManager, console: Console) -> None:
    """Display results summary"""
    if not results:
        console.print("[red]No results to display[/red]")
        return
    
    summary = rollout_manager.get_rollout_summary()
    
    # Overall statistics
    stats_table = Table(title="Rollout Results Summary")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Episodes", str(summary["total_episodes"]))
    stats_table.add_row("Success Rate", f"{summary['success_rate']:.1%}")
    stats_table.add_row("Average Duration", f"{summary['avg_duration']:.2f}s")
    stats_table.add_row("Total Duration", f"{summary['total_duration']:.2f}s")
    
    # Add token usage statistics if available
    if any(result.token_usage for result in results):
        total_input_tokens = sum(result.token_usage.get("total_input_tokens", 0) for result in results if result.token_usage)
        total_output_tokens = sum(result.token_usage.get("total_output_tokens", 0) for result in results if result.token_usage)
        total_api_calls = sum(result.token_usage.get("api_calls", 0) for result in results if result.token_usage)
        
        stats_table.add_row("Total Input Tokens", f"{total_input_tokens:,}")
        stats_table.add_row("Total Output Tokens", f"{total_output_tokens:,}")
        stats_table.add_row("Total API Calls", str(total_api_calls))
    
    console.print(stats_table)
    console.print()
    
    # Environment-specific results
    if "environments" in summary:
        env_table = Table(title="Environment Results")
        env_table.add_column("Environment", style="cyan")
        env_table.add_column("Episodes", style="green")
        env_table.add_column("Success Rate", style="green")
        env_table.add_column("Avg Score", style="green")
        env_table.add_column("Avg Duration", style="green")
        
        for env_id, env_stats in summary["environments"].items():
            success_rate = env_stats["successful"] / env_stats["episodes"] if env_stats["episodes"] > 0 else 0
            avg_duration = env_stats["total_duration"] / env_stats["episodes"] if env_stats["episodes"] > 0 else 0
            
            env_table.add_row(
                env_id,
                str(env_stats["episodes"]),
                f"{success_rate:.1%}",
                f"{env_stats['avg_score']:.2f}",
                f"{avg_duration:.2f}s"
            )
        
        console.print(env_table)
    
    # Token usage and cache statistics
    if any(result.token_usage for result in results):
        cache_hits = sum(result.token_usage.get("cache_hits", 0) for result in results if result.token_usage)
        cache_misses = sum(result.token_usage.get("cache_misses", 0) for result in results if result.token_usage)
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        cache_panel = Panel(
            f"[bold]Cache Hits:[/bold] {cache_hits:,}\n"
            f"[bold]Cache Misses:[/bold] {cache_misses:,}\n"
            f"[bold]Cache Hit Rate:[/bold] {cache_hit_rate:.1%}\n"
            f"[bold]Total Requests:[/bold] {cache_hits + cache_misses:,}",
            title="Cache Statistics",
            border_style="blue"
        )
        console.print(cache_panel)
        console.print()
    
    # Plugin statistics
    if "plugin_stats" in summary:
        plugin_stats = summary["plugin_stats"]
        plugin_panel = Panel(
            f"[bold]Total Plugins:[/bold] {plugin_stats['total_plugins']}\n"
            f"[bold]Enabled:[/bold] {plugin_stats['enabled_plugins']}\n"
            f"[bold]Disabled:[/bold] {plugin_stats['disabled_plugins']}",
            title="Plugin Statistics",
            border_style="magenta"
        )
        console.print(plugin_panel)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM RL Framework")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--output", default="results.json",
                       help="Output file for results")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate configuration without running")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output except errors")
    
    # LLM configuration arguments
    parser.add_argument("--llm-model", default="anthropic/claude-sonnet-4",
                       help="LLM model to use")
    parser.add_argument("--llm-api-key", required=True,
                       help="API key for the LLM service")
    parser.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1",
                       help="Base URL for the LLM API")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                       help="Temperature for LLM sampling")
    parser.add_argument("--llm-max-tokens", type=int, default=4096,
                       help="Maximum tokens for LLM response")
    parser.add_argument("--llm-timeout", type=int, default=60,
                       help="Timeout for LLM API calls in seconds")
    
    # Token caching and context management options
    parser.add_argument("--llm-enable-caching", action="store_true", default=True,
                       help="Enable response caching to reduce API calls")
    parser.add_argument("--llm-disable-caching", action="store_true",
                       help="Disable response caching")
    parser.add_argument("--llm-cache-size", type=int, default=100,
                       help="Maximum number of cached responses")
    parser.add_argument("--llm-max-context-messages", type=int, default=50,
                       help="Maximum messages in context before truncation")
    parser.add_argument("--llm-max-output-length", type=int, default=2000,
                       help="Maximum length of command output to include in context")
    parser.add_argument("--llm-disable-token-tracking", action="store_true",
                       help="Disable token usage tracking")
    parser.add_argument("--llm-disable-high-usage-warnings", action="store_true",
                       help="Disable high usage warnings")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    console = Console()
    
    if not args.quiet:
        console.print(Panel(
            "[bold blue]Real Work - Ritser Labs[/bold blue]\n"
            "Extensible framework for reinforcement learning with LLM agents",
            title="Welcome",
            border_style="blue"
        ))
    
    try:
        # Load configuration
        if not args.quiet:
            console.print("[bold]Loading configuration...[/bold]")
        
        config = load_config(args.config)
        
        # Create LLM config from command-line arguments
        config.llm_config = LLMConfig(
            model=args.llm_model,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
            # Token caching and context management
            enable_caching=args.llm_enable_caching and not args.llm_disable_caching,
            cache_size=args.llm_cache_size,
            max_context_messages=args.llm_max_context_messages,
            max_output_length=args.llm_max_output_length,
            # Token usage tracking
            track_token_usage=not args.llm_disable_token_tracking,
            warn_high_usage=not args.llm_disable_high_usage_warnings
        )
        
        if not args.quiet:
            display_config_summary(config, console)
        
        # Validate configuration
        if not config.environments:
            console.print("[red]Error: No environments specified in configuration[/red]")
            sys.exit(1)
        
        if args.dry_run:
            console.print("[green]Configuration validation passed![/green]")
            return
        
        # Initialize rollout manager
        if not args.quiet:
            console.print("[bold]Initializing rollout manager...[/bold]")
        
        rollout_manager = RolloutManager(config)
        
        if not await rollout_manager.initialize():
            console.print("[red]Error: Failed to initialize rollout manager[/red]")
            sys.exit(1)
        
        try:
            # Run rollouts
            if not args.quiet:
                console.print("[bold]Starting rollouts...[/bold]")
            
            results = await rollout_manager.run_rollouts()
            
            # Display results
            if not args.quiet:
                console.print("[bold]Rollouts completed![/bold]")
                display_results_summary(results, rollout_manager, console)
            
            # Export results
            if not args.quiet:
                console.print(f"[bold]Exporting results to {args.output}...[/bold]")
            
            await rollout_manager.export_results(args.output)
            
            if not args.quiet:
                console.print(f"[green]Results exported successfully![/green]")
                console.print(f"[blue]Results saved to: {args.output}[/blue]")
                console.print(f"[blue]Statistics saved to: {Path(args.output).with_suffix('.stats.json')}[/blue]")
        
        finally:
            # Cleanup
            await rollout_manager.cleanup()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.log_level == "DEBUG":
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 