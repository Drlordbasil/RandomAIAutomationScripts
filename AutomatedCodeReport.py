import os
import re
import ast
import subprocess
from collections import defaultdict
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

class EnhancedLocalCodeAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.python_files = []
        self.stats = defaultdict(int)
        self.large_functions = []
        self.complex_functions = []
        self.todos = []
        self.imports = defaultdict(set)
        self.dependency_graph = nx.DiGraph()
        self.class_info = defaultdict(dict)
        self.function_info = defaultdict(dict)

    def scan_directory(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
                    self.python_files.append(os.path.join(root, file))

    def analyze_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            self.stats['total_files'] += 1
            self.stats['total_lines'] += len(content.splitlines())
            self.analyze_imports(content, file_path)
            self.analyze_todos(content, file_path)
            self.analyze_ast(content, file_path)

    def analyze_imports(self, content, file_path):
        import_pattern = r'^import\s+(\w+)|^from\s+(\w+)\s+import'
        matches = re.finditer(import_pattern, content, re.MULTILINE)
        for match in matches:
            module = match.group(1) or match.group(2)
            self.imports[module].add(file_path)
            self.dependency_graph.add_edge(os.path.basename(file_path), module)

    def analyze_todos(self, content, file_path):
        todo_pattern = r'#\s*TODO:?\s*(.+)$'
        matches = re.finditer(todo_pattern, content, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            self.todos.append((file_path, match.group(1).strip()))

    def analyze_ast(self, content, file_path):
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.analyze_function(node, file_path)
                elif isinstance(node, ast.ClassDef):
                    self.analyze_class(node, file_path)
        except SyntaxError:
            self.stats['files_with_syntax_errors'] += 1

    def analyze_function(self, node, file_path):
        self.stats['total_functions'] += 1
        complexity = self.calculate_complexity(node)
        self.function_info[file_path][node.name] = {
            'lines': len(node.body),
            'complexity': complexity,
            'arguments': len(node.args.args)
        }
        if len(node.body) > 20:
            self.large_functions.append((file_path, node.name, len(node.body)))
        if complexity > 10:
            self.complex_functions.append((file_path, node.name, complexity))

    def analyze_class(self, node, file_path):
        self.stats['total_classes'] += 1
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        self.class_info[file_path][node.name] = {
            'methods': methods,
            'method_count': len(methods)
        }

    def calculate_complexity(self, node):
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def generate_dependency_graph(self):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.dependency_graph)
        nx.draw(self.dependency_graph, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, font_weight='bold')
        plt.title("Module Dependency Graph")
        graph_file = 'dependency_graph.png'
        plt.savefig(graph_file)
        plt.close()
        return graph_file

    def generate_report(self):
        report = f"Enhanced Code Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "General Statistics:\n"
        report += f"Total Python files: {self.stats['total_files']}\n"
        report += f"Total lines of code: {self.stats['total_lines']}\n"
        report += f"Total functions: {self.stats['total_functions']}\n"
        report += f"Total classes: {self.stats['total_classes']}\n"
        report += f"Files with syntax errors: {self.stats['files_with_syntax_errors']}\n\n"

        report += "Large Functions (>20 lines):\n"
        for file_path, func_name, lines in sorted(self.large_functions, key=lambda x: x[2], reverse=True)[:10]:
            report += f"- {os.path.basename(file_path)}: {func_name} ({lines} lines)\n"
        report += "\n"

        report += "Complex Functions (Cyclomatic Complexity >10):\n"
        for file_path, func_name, complexity in sorted(self.complex_functions, key=lambda x: x[2], reverse=True)[:10]:
            report += f"- {os.path.basename(file_path)}: {func_name} (Complexity: {complexity})\n"
        report += "\n"

        if self.todos:
            report += "TODOs:\n"
            for file_path, todo in self.todos[:20]:
                report += f"- {os.path.basename(file_path)}: {todo}\n"
            report += "\n"
        else:
            report += "No TODOs found in the project.\n\n"

        report += "Most Common Imports:\n"
        common_imports = sorted(self.imports.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for module, files in common_imports:
            report += f"- {module}: Used in {len(files)} files\n"
        report += "\n"

        report += "Class Information:\n"
        for file_path, classes in self.class_info.items():
            report += f"File: {os.path.basename(file_path)}\n"
            for class_name, info in classes.items():
                report += f"  - Class: {class_name}\n"
                report += f"    Methods: {', '.join(info['methods'])}\n"
                report += f"    Method Count: {info['method_count']}\n"
        report += "\n"

        report += "Function Complexity Distribution:\n"
        complexity_dist = defaultdict(int)
        for file_funcs in self.function_info.values():
            for func_info in file_funcs.values():
                complexity_dist[func_info['complexity']] += 1
        for complexity, count in sorted(complexity_dist.items()):
            report += f"  Complexity {complexity}: {count} functions\n"

        return report

    def save_report(self, report):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_code_analysis_report_{timestamp}.html"
        dependency_graph = self.generate_dependency_graph()
        html_report = f"""
        <html>
        <head>
            <title>Enhanced Code Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; }}
                .graph {{ text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Enhanced Code Analysis Report</h1>
            <pre>{report}</pre>
            <div class="graph">
                <h2>Module Dependency Graph</h2>
                <img src="{dependency_graph}" alt="Module Dependency Graph">
            </div>
        </body>
        </html>
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_report)
        return filename

    def open_in_chrome(self, filename):
        try:
            chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
            if os.path.exists(chrome_path):
                subprocess.run([chrome_path, os.path.abspath(filename)])
            else:
                print(f"Report saved as {filename}, but Chrome not found at the expected location.")
        except Exception as e:
            print(f"Error opening Chrome: {e}")

    def run(self):
        print("Starting enhanced code analysis...")
        self.scan_directory()
        for file in self.python_files:
            self.analyze_file(file)
        report = self.generate_report()
        report_file = self.save_report(report)
        print(f"Analysis complete. Opening report in Chrome...")
        self.open_in_chrome(report_file)

if __name__ == "__main__":
    root_directory = os.getcwd()  # Use current directory, or specify a path
    analyzer = EnhancedLocalCodeAnalyzer(root_directory)
    analyzer.run()
