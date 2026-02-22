from llm.async_llm import create_llm_instance
from llm.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from typing import Optional

class BaseAgent:
    def __init__(self, role, system_prompt, llm_config):
        self.llm = create_llm_instance(llm_config, system_prompt)
        self.llm_config = llm_config
        self.role = role
    
    async def _fill_node(self, op_class, operator, prompt, mode=None, memory=None, **extra_kwargs):
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode)
        
        try:
            # Use the formatter with AsyncLLM
            if formatter:
                response = await self.llm.call_with_format(operator, prompt, formatter, memory)
            else:
                # Fallback to direct call if no formatter is needed
                response = await self.llm(operator, prompt)
                
            # Convert to expected format based on the original implementation
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            return {"error": str(e)}
    
    def _create_formatter(self, op_class, mode=None) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            return CodeFormatter()
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None