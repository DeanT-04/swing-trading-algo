#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy optimization module.

This module provides functionality to optimize strategy parameters
using various methods such as grid search, random search, or genetic algorithms.
"""

import logging
import itertools
import random
from typing import Dict, List, Tuple, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.models import Stock, TimeFrame
from src.strategy.basic_swing import BasicSwingStrategy
from src.simulation.simulator import TradeSimulator
from src.performance.analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizes strategy parameters using various methods.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the strategy optimizer with configuration parameters.
        
        Args:
            config: Optimizer configuration dictionary
        """
        self.config = config
        
        # Extract configuration parameters
        self.method = config.get("method", "grid_search")
        self.parameters = config.get("parameters", [])
        self.metric = config.get("metric", "sharpe_ratio")
        self.population_size = config.get("population_size", 20)
        self.generations = config.get("generations", 5)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.random_trials = config.get("random_trials", 20)
        
        logger.info(f"Initialized StrategyOptimizer with method: {self.method}")
    
    def optimize(self, stocks: Dict[str, Stock], timeframe: TimeFrame, 
                initial_balance: float, parameter_ranges: Dict[str, List[Any]]) -> Dict:
        """
        Optimize strategy parameters using the specified method.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            initial_balance: Initial account balance
            parameter_ranges: Dictionary of parameter names and their possible values
            
        Returns:
            Dict: Dictionary of optimized parameters and their performance
        """
        if self.method == "grid_search":
            return self._grid_search(stocks, timeframe, initial_balance, parameter_ranges)
        elif self.method == "random_search":
            return self._random_search(stocks, timeframe, initial_balance, parameter_ranges)
        elif self.method == "genetic":
            return self._genetic_algorithm(stocks, timeframe, initial_balance, parameter_ranges)
        else:
            raise ValueError(f"Unsupported optimization method: {self.method}")
    
    def _evaluate_parameters(self, stocks: Dict[str, Stock], timeframe: TimeFrame, 
                           initial_balance: float, parameters: Dict[str, Any]) -> float:
        """
        Evaluate a set of parameters by running a simulation and calculating the performance metric.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            initial_balance: Initial account balance
            parameters: Dictionary of parameter values to evaluate
            
        Returns:
            float: Performance metric value
        """
        # Create strategy with the parameters
        strategy_config = parameters.copy()
        strategy = BasicSwingStrategy(strategy_config)
        
        # Create simulator
        simulator_config = {
            "initial_balance": initial_balance,
            "currency": "GBP",
            "risk_per_trade": 0.02,
            "max_open_positions": 3,
            "slippage": 0.001,
            "commission": 0.002,
            "enable_fractional_shares": True
        }
        simulator = TradeSimulator(simulator_config)
        
        # Run simulation
        end_date = datetime.now()
        
        # Analyze each stock and generate signals
        for symbol, stock in stocks.items():
            # Analyze the stock
            results = strategy.analyze(stock, timeframe)
            
            if not results or "signals" not in results or not results["signals"]:
                continue
            
            # Process signals
            for signal in results["signals"]:
                # Add symbol to signal
                signal["symbol"] = symbol
                
                # Get the timestamp and corresponding price data
                timestamp = signal["timestamp"]
                df = stock.get_dataframe(timeframe)
                if timestamp not in df.index:
                    continue
                
                current_price = df.loc[timestamp, "close"]
                
                # Process the signal
                simulator.process_signal(signal, stock, current_price, timestamp)
        
        # Update positions for all stocks
        simulator.update_positions(stocks, end_date)
        
        # Get trade history
        trade_history = simulator.get_trade_history()
        
        # Calculate performance metrics
        analyzer = PerformanceAnalyzer({"metrics": [self.metric]})
        metrics = analyzer.analyze_trades(trade_history, initial_balance)
        
        # Return the specified metric
        if self.metric == "sharpe_ratio":
            return metrics.get("sharpe_ratio", 0.0)
        elif self.metric == "profit_factor":
            return metrics.get("profit_factor", 0.0)
        elif self.metric == "win_rate":
            return metrics.get("win_rate", 0.0)
        elif self.metric == "total_return":
            return metrics.get("total_return", 0.0)
        elif self.metric == "total_return_percent":
            return metrics.get("total_return_percent", 0.0)
        elif self.metric == "max_drawdown":
            # Invert max_drawdown so that lower values are better
            return -metrics.get("max_drawdown_percent", 0.0)
        else:
            return 0.0
    
    def _grid_search(self, stocks: Dict[str, Stock], timeframe: TimeFrame, 
                   initial_balance: float, parameter_ranges: Dict[str, List[Any]]) -> Dict:
        """
        Optimize parameters using grid search.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            initial_balance: Initial account balance
            parameter_ranges: Dictionary of parameter names and their possible values
            
        Returns:
            Dict: Dictionary of optimized parameters and their performance
        """
        logger.info("Starting grid search optimization")
        
        # Get parameter names and values
        param_names = list(parameter_ranges.keys())
        param_values = [parameter_ranges[name] for name in param_names]
        
        # Generate all combinations of parameters
        combinations = list(itertools.product(*param_values))
        
        # Evaluate each combination
        best_score = float('-inf')
        best_params = None
        
        for i, combo in enumerate(combinations):
            # Create parameter dictionary
            params = {name: value for name, value in zip(param_names, combo)}
            
            # Evaluate parameters
            score = self._evaluate_parameters(stocks, timeframe, initial_balance, params)
            
            logger.info(f"Combination {i+1}/{len(combinations)}: {params}, Score: {score}")
            
            # Update best parameters if score is better
            if score > best_score:
                best_score = score
                best_params = params
        
        logger.info(f"Grid search completed. Best parameters: {best_params}, Score: {best_score}")
        
        return {
            "parameters": best_params,
            "score": best_score,
            "method": "grid_search"
        }
    
    def _random_search(self, stocks: Dict[str, Stock], timeframe: TimeFrame, 
                     initial_balance: float, parameter_ranges: Dict[str, List[Any]]) -> Dict:
        """
        Optimize parameters using random search.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            initial_balance: Initial account balance
            parameter_ranges: Dictionary of parameter names and their possible values
            
        Returns:
            Dict: Dictionary of optimized parameters and their performance
        """
        logger.info("Starting random search optimization")
        
        # Get parameter names
        param_names = list(parameter_ranges.keys())
        
        # Evaluate random combinations
        best_score = float('-inf')
        best_params = None
        
        for i in range(self.random_trials):
            # Generate random parameters
            params = {}
            for name in param_names:
                params[name] = random.choice(parameter_ranges[name])
            
            # Evaluate parameters
            score = self._evaluate_parameters(stocks, timeframe, initial_balance, params)
            
            logger.info(f"Trial {i+1}/{self.random_trials}: {params}, Score: {score}")
            
            # Update best parameters if score is better
            if score > best_score:
                best_score = score
                best_params = params
        
        logger.info(f"Random search completed. Best parameters: {best_params}, Score: {best_score}")
        
        return {
            "parameters": best_params,
            "score": best_score,
            "method": "random_search"
        }
    
    def _genetic_algorithm(self, stocks: Dict[str, Stock], timeframe: TimeFrame, 
                         initial_balance: float, parameter_ranges: Dict[str, List[Any]]) -> Dict:
        """
        Optimize parameters using a genetic algorithm.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            initial_balance: Initial account balance
            parameter_ranges: Dictionary of parameter names and their possible values
            
        Returns:
            Dict: Dictionary of optimized parameters and their performance
        """
        logger.info("Starting genetic algorithm optimization")
        
        # Get parameter names
        param_names = list(parameter_ranges.keys())
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = {}
            for name in param_names:
                individual[name] = random.choice(parameter_ranges[name])
            population.append(individual)
        
        # Evaluate initial population
        scores = []
        for individual in population:
            score = self._evaluate_parameters(stocks, timeframe, initial_balance, individual)
            scores.append(score)
        
        # Run genetic algorithm
        for generation in range(self.generations):
            logger.info(f"Generation {generation+1}/{self.generations}")
            
            # Select parents
            parents = self._select_parents(population, scores)
            
            # Create offspring
            offspring = self._create_offspring(parents, parameter_ranges)
            
            # Evaluate offspring
            offspring_scores = []
            for individual in offspring:
                score = self._evaluate_parameters(stocks, timeframe, initial_balance, individual)
                offspring_scores.append(score)
            
            # Replace population
            population = offspring
            scores = offspring_scores
        
        # Find best individual
        best_index = scores.index(max(scores))
        best_params = population[best_index]
        best_score = scores[best_index]
        
        logger.info(f"Genetic algorithm completed. Best parameters: {best_params}, Score: {best_score}")
        
        return {
            "parameters": best_params,
            "score": best_score,
            "method": "genetic"
        }
    
    def _select_parents(self, population: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population: List of individuals (parameter dictionaries)
            scores: List of scores for each individual
            
        Returns:
            List[Dict[str, Any]]: Selected parents
        """
        parents = []
        
        # Tournament selection
        for _ in range(len(population)):
            # Select two random individuals
            i1 = random.randint(0, len(population) - 1)
            i2 = random.randint(0, len(population) - 1)
            
            # Select the better individual
            if scores[i1] > scores[i2]:
                parents.append(population[i1])
            else:
                parents.append(population[i2])
        
        return parents
    
    def _create_offspring(self, parents: List[Dict[str, Any]], parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Create offspring through crossover and mutation.
        
        Args:
            parents: List of parent individuals
            parameter_ranges: Dictionary of parameter names and their possible values
            
        Returns:
            List[Dict[str, Any]]: Offspring individuals
        """
        offspring = []
        
        # Create pairs of parents
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Perform crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Perform mutation
                child1 = self._mutate(child1, parameter_ranges)
                child2 = self._mutate(child2, parameter_ranges)
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # If there's an odd number of parents, just add the last one
                offspring.append(parents[i].copy())
        
        return offspring
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Two child individuals
        """
        child1 = {}
        child2 = {}
        
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        param_names = list(parent1.keys())
        
        for i, name in enumerate(param_names):
            if i < crossover_point:
                child1[name] = parent1[name]
                child2[name] = parent2[name]
            else:
                child1[name] = parent2[name]
                child2[name] = parent1[name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Perform mutation on an individual.
        
        Args:
            individual: Individual to mutate
            parameter_ranges: Dictionary of parameter names and their possible values
            
        Returns:
            Dict[str, Any]: Mutated individual
        """
        mutated = individual.copy()
        
        for name in mutated:
            if random.random() < self.mutation_rate:
                mutated[name] = random.choice(parameter_ranges[name])
        
        return mutated
