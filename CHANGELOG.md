# Changelog

## Version 2.0.0 - Tool Calling Update

### 🎉 Major Updates

#### **Native OpenAI Tool Calling**
- **Replaced manual XML parsing with OpenAI's native tool calling functionality**
- **Removed ActionParser class** - no longer needed with native tool calling
- **Added 4 core tools** for the LLM:
  - `execute_command` - Execute shell commands with optional timeout and working directory
  - `write_file` - Write content to files with filepath and content parameters
  - `read_file` - Read content from files with filepath parameter
  - `mark_done` - Mark task as completed with optional message

#### **OpenRouter Integration**
- **Updated to use OpenRouter API** instead of direct OpenAI API
- **Added support for Claude 4 Sonnet** (`anthropic/claude-4-sonnet`)
- **Environment variable support** - automatically uses `OPENROUTER_API_KEY` if available
- **Backward compatibility** - still supports direct OpenAI API configuration

#### **Enhanced Configuration**
- **Updated example configuration** to use Claude 3.5 Sonnet
- **Changed default base URL** to OpenRouter (`https://openrouter.ai/api/v1`)
- **API key validation** - better error messages for missing/invalid keys

#### **Improved Testing**
- **Updated test framework** to test tool calling functionality
- **Added LLM agent tool tests** - validates tool creation and mapping
- **Created tool calling demo script** (`test_tool_calling.py`)
- **All tests passing** - comprehensive validation of new functionality

### 🔧 Technical Improvements

#### **LLMAgent Class Refactoring**
- **Removed manual response parsing** - now uses OpenAI's tool calling
- **Added tool definitions** - structured JSON schema for each tool
- **Improved conversation handling** - proper tool call result integration
- **Better error handling** - graceful fallback for tool call failures

#### **Action Mapping**
- **Tool-to-action conversion** - seamless integration with existing action system
- **Preserved existing action types** - no breaking changes to core framework
- **Enhanced action creation** - more robust parameter handling

### 📚 Documentation Updates

#### **Updated README**
- **New tool calling section** - explains native function calling
- **OpenRouter setup instructions** - environment variable configuration
- **Updated examples** - reflects new Claude model and tool calling
- **Enhanced features list** - highlights tool calling improvements

#### **Configuration Examples**
- **Updated calculator API config** - uses Claude 3.5 Sonnet
- **New configuration format** - reflects OpenRouter integration
- **Better documentation** - clearer setup instructions

### 🛠 Files Modified

#### **Core Framework**
- `src/agents/llm_agent.py` - Complete rewrite for tool calling
- `examples/calculator_api/config.json` - Updated to use Claude model
- `test_framework.py` - Updated tests for tool calling
- `README.md` - Updated documentation

#### **New Files**
- `test_tool_calling.py` - Demonstration script for tool calling
- `CHANGELOG.md` - This changelog

### 🚀 Benefits

1. **More Reliable** - Native tool calling is more robust than text parsing
2. **Better Performance** - Direct function calls instead of regex parsing
3. **Easier to Extend** - Adding new tools is straightforward
4. **Industry Standard** - Uses OpenAI's standard tool calling format
5. **Advanced Model** - Claude 3.5 Sonnet provides better coding capabilities
6. **Cost Effective** - OpenRouter provides competitive pricing

### 🧪 Testing

All existing functionality has been preserved and enhanced:
- ✅ Configuration loading and validation
- ✅ Trajectory management and persistence
- ✅ Plugin system functionality
- ✅ Tool calling with proper action mapping
- ✅ Example configuration validation
- ✅ Framework integration tests

### 📋 Next Steps

To use the updated framework:

1. **Set OpenRouter API key**: `export OPENROUTER_API_KEY="your_key"`
2. **Test the framework**: `python test_framework.py`
3. **Run dry-run**: `python main.py examples/calculator_api/config.json --dry-run`
4. **Execute rollout**: `python main.py examples/calculator_api/config.json`

The framework is now ready for production use with state-of-the-art LLM capabilities! 