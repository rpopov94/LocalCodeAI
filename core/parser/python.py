"""Python parser."""
import ast
from typing import List, Dict
from pathlib import Path


class PythonCodeParser:
    """Ast tree pyparser."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def parse_file(self, file_path: Path) -> List[Dict]:
        """Parse one file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                if not code.strip():
                    return []

                tree = ast.parse(code)
                if not isinstance(tree, ast.Module):
                    return []

        except (SyntaxError, UnicodeDecodeError, AttributeError) as e:
            print(f"Skipping {file_path}: {str(e)}")
            return []

        entities = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node) or ""
                entities.append({
                    'type': type(node).__name__,
                    'name': node.name,
                    'docstring': docstring,
                    'code': ast.unparse(node),
                    'file': str(file_path.relative_to(self.repo_path))
                })
        return entities

    def parse_project(self) -> List[Dict]:
        """Recursive parsing of the entire project."""
        all_entities = []
        for py_file in self.repo_path.rglob('*.py'):
            all_entities.extend(self.parse_file(py_file))
        return all_entities
