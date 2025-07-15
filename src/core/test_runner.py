import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .config import TestResult, EnvironmentConfig, TimeoutConfig
from ..environments.environment import Environment


class UnitTestRunner:
    """Manages unit test execution and result collection"""
    
    def __init__(self, timeout_config: TimeoutConfig = None):
        self.timeout_config = timeout_config or TimeoutConfig()
        self.logger = logging.getLogger("UnitTestRunner")
    
    async def run_tests(self, environment: Environment, test_commands: List[str] = None) -> List[TestResult]:
        """Run unit tests in the environment"""
        if test_commands is None:
            test_commands = environment.config.unit_tests
        
        test_results = []
        
        self.logger.info(f"Running {len(test_commands)} tests in environment {environment.config.id}")
        
        for i, test_command in enumerate(test_commands):
            self.logger.info(f"Running test {i+1}/{len(test_commands)}: {test_command}")
            
            start_time = time.time()
            
            try:
                # Execute test command
                exec_result = environment.container.exec_run(
                    test_command,
                    workdir=environment.config.working_directory,
                    user="1000:1000",
                    environment=environment.config.environment_variables,
                    stdout=True,
                    stderr=True,
                    detach=False
                )
                
                output = exec_result.output.decode('utf-8', errors='ignore')
                exit_code = exec_result.exit_code
                success = exit_code == 0
                
                test_result = TestResult(
                    command=test_command,
                    success=success,
                    output=output,
                    error=output if not success else "",
                    exit_code=exit_code,
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
                
                test_results.append(test_result)
                
                self.logger.info(f"Test {'PASSED' if success else 'FAILED'}: {test_command}")
                
            except Exception as e:
                self.logger.error(f"Error running test {test_command}: {e}")
                
                test_result = TestResult(
                    command=test_command,
                    success=False,
                    output="",
                    error=str(e),
                    exit_code=-1,
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
                
                test_results.append(test_result)
        
        return test_results
    
    async def run_single_test(self, environment: Environment, test_command: str) -> TestResult:
        """Run a single test command"""
        results = await self.run_tests(environment, [test_command])
        return results[0] if results else None
    
    def calculate_test_score(self, test_results: List[TestResult]) -> float:
        """Calculate overall test score (0.0 to 1.0)"""
        if not test_results:
            return 0.0
        
        passed_tests = sum(1 for test in test_results if test.success)
        total_tests = len(test_results)
        
        return passed_tests / total_tests
    
    def get_test_summary(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Get a summary of test results"""
        if not test_results:
            return {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "success_rate": 0.0,
                "total_duration": 0.0,
                "avg_duration": 0.0
            }
        
        passed = sum(1 for test in test_results if test.success)
        failed = len(test_results) - passed
        total_duration = sum(test.duration for test in test_results)
        
        return {
            "total_tests": len(test_results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_results),
            "total_duration": total_duration,
            "avg_duration": total_duration / len(test_results),
            "failed_tests": [
                {"command": test.command, "error": test.error}
                for test in test_results if not test.success
            ]
        }
    
    def get_detailed_test_report(self, test_results: List[TestResult]) -> str:
        """Generate a detailed test report"""
        if not test_results:
            return "No tests were run."
        
        report = []
        summary = self.get_test_summary(test_results)
        
        report.append("=" * 50)
        report.append("TEST RESULTS SUMMARY")
        report.append("=" * 50)
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed']}")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Success Rate: {summary['success_rate']:.2%}")
        report.append(f"Total Duration: {summary['total_duration']:.2f}s")
        report.append(f"Average Duration: {summary['avg_duration']:.2f}s")
        report.append("")
        
        if summary['failed_tests']:
            report.append("FAILED TESTS:")
            report.append("-" * 20)
            for i, failed_test in enumerate(summary['failed_tests'], 1):
                report.append(f"{i}. {failed_test['command']}")
                report.append(f"   Error: {failed_test['error'][:200]}...")
                report.append("")
        
        report.append("DETAILED RESULTS:")
        report.append("-" * 20)
        
        for i, test in enumerate(test_results, 1):
            status = "PASS" if test.success else "FAIL"
            report.append(f"{i}. [{status}] {test.command} ({test.duration:.2f}s)")
            
            if not test.success:
                report.append(f"   Error: {test.error[:200]}...")
            
            report.append("")
        
        return "\n".join(report)
    
    async def run_tests_with_timeout(self, environment: Environment, 
                                   test_commands: List[str] = None,
                                   timeout: int = None) -> List[TestResult]:
        """Run tests with a global timeout"""
        timeout = timeout or self.timeout_config.test_timeout
        
        try:
            return await asyncio.wait_for(
                self.run_tests(environment, test_commands),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Test execution timed out after {timeout} seconds")
            
            # Return a failed test result for timeout
            return [TestResult(
                command="ALL_TESTS",
                success=False,
                output="",
                error=f"Test execution timed out after {timeout} seconds",
                exit_code=-1,
                duration=timeout,
                timestamp=datetime.now().isoformat()
            )]
    
    async def run_tests_parallel(self, environments: List[Environment], 
                               test_commands: List[str] = None) -> Dict[str, List[TestResult]]:
        """Run tests in parallel across multiple environments"""
        tasks = []
        env_ids = []
        
        for env in environments:
            task = self.run_tests(env, test_commands)
            tasks.append(task)
            env_ids.append(env.config.id)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        test_results = {}
        for env_id, result in zip(env_ids, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error running tests in {env_id}: {result}")
                test_results[env_id] = []
            else:
                test_results[env_id] = result
        
        return test_results
    
    def compare_test_results(self, before: List[TestResult], after: List[TestResult]) -> Dict[str, Any]:
        """Compare test results to identify improvements or regressions"""
        before_by_command = {test.command: test for test in before}
        after_by_command = {test.command: test for test in after}
        
        improvements = []
        regressions = []
        unchanged = []
        
        all_commands = set(before_by_command.keys()) | set(after_by_command.keys())
        
        for command in all_commands:
            before_test = before_by_command.get(command)
            after_test = after_by_command.get(command)
            
            if before_test and after_test:
                if not before_test.success and after_test.success:
                    improvements.append(command)
                elif before_test.success and not after_test.success:
                    regressions.append(command)
                else:
                    unchanged.append(command)
            elif before_test and not after_test:
                # Test was removed
                pass
            elif not before_test and after_test:
                # Test was added
                if after_test.success:
                    improvements.append(command)
                else:
                    regressions.append(command)
        
        return {
            "improvements": improvements,
            "regressions": regressions,
            "unchanged": unchanged,
            "improvement_count": len(improvements),
            "regression_count": len(regressions),
            "net_change": len(improvements) - len(regressions)
        } 