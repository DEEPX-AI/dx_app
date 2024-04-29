#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"


using namespace rapidjson;

typedef GenericValue<UTF8<>, CrtAllocator > ValueType;

namespace dxapp
{
// Forward ref
static void CreateErrorMessages(const ValueType& errors, size_t depth, const char* context);

// Convert GenericValue to std::string
static std::string GetString(const ValueType& val) {
  std::ostringstream s;
  if (val.IsString())
    s << val.GetString();
  else if (val.IsDouble())
    s << val.GetDouble();
  else if (val.IsUint())
   s << val.GetUint();
  else if (val.IsInt())
    s << val.GetInt();
  else if (val.IsUint64())
    s << val.GetUint64();
  else if (val.IsInt64())
    s <<  val.GetInt64();
  else if (val.IsBool() && val.GetBool())
    s << "true";
  else if (val.IsBool())
    s << "false";
  else if (val.IsFloat())
    s << val.GetFloat();
  return s.str();
}

// Create the error message for a named error
// The error object can either be empty or contain at least member properties:
// {"errorCode": <code>, "instanceRef": "<pointer>", "schemaRef": "<pointer>" }
// Additional properties may be present for use as inserts.
// An "errors" property may be present if there are child errors.
static void HandleError(const char* errorName, const ValueType& error, size_t depth, const char* context) {
  if (!error.ObjectEmpty()) {
    // Get error code and look up error message text (English)
    int code = error["errorCode"].GetInt();
    std::string message(GetValidateError_En(static_cast<ValidateErrorCode>(code)));
    // For each member property in the error, see if its name exists as an insert in the error message and if so replace with the stringified property value
    // So for example - "Number '%actual' is not a multiple of the 'multipleOf' value '%expected'." - we would expect "actual" and "expected" members.
    for (ValueType::ConstMemberIterator insertsItr = error.MemberBegin();
      insertsItr != error.MemberEnd(); ++insertsItr) {
      std::string insertName("%");
      insertName += insertsItr->name.GetString(); // eg "%actual"
      size_t insertPos = message.find(insertName);
      if (insertPos != std::string::npos) {
        std::string insertString("");
        const ValueType &insert = insertsItr->value;
        if (insert.IsArray()) {
          // Member is an array so create comma-separated list of items for the insert string
          for (ValueType::ConstValueIterator itemsItr = insert.Begin(); itemsItr != insert.End(); ++itemsItr) {
            if (itemsItr != insert.Begin()) insertString += ",";
            insertString += GetString(*itemsItr);
          }
        } else {
          insertString += GetString(insert);
        }
        message.replace(insertPos, insertName.length(), insertString);
      }
    }
    // Output error message, references, context
    std::string indent(depth * 2, ' ');
    std::cout << indent << "Error Name: " << errorName << std::endl;
    std::cout << indent << "Message: " << message.c_str() << std::endl;
    std::cout << indent << "Instance: " << error["instanceRef"].GetString() << std::endl;
    std::cout << indent << "Schema: " << error["schemaRef"].GetString() << std::endl;
    if (depth > 0) std::cout << indent << "Context: " << context << std::endl;
    std::cout << std::endl;

    // If child errors exist, apply the process recursively to each error structure.
    // This occurs for "oneOf", "allOf", "anyOf" and "dependencies" errors, so pass the error name as context.
    if (error.HasMember("errors")) {
      depth++;
      const ValueType &childErrors = error["errors"];
      if (childErrors.IsArray()) {
        // Array - each item is an error structure - example
        // "anyOf": {"errorCode": ..., "errors":[{"pattern": {"errorCode\": ...\"}}, {"pattern": {"errorCode\": ...}}]
        for (ValueType::ConstValueIterator errorsItr = childErrors.Begin();
             errorsItr != childErrors.End(); ++errorsItr) {
          CreateErrorMessages(*errorsItr, depth, errorName);
        }
      } else if (childErrors.IsObject()) {
        // Object - each member is an error structure - example
        // "dependencies": {"errorCode": ..., "errors": {"address": {"required": {"errorCode": ...}}, "name": {"required": {"errorCode": ...}}}
        for (ValueType::ConstMemberIterator propsItr = childErrors.MemberBegin();
             propsItr != childErrors.MemberEnd(); ++propsItr) {
          CreateErrorMessages(propsItr->value, depth, errorName);
        }
      }
    }
  }
}

// Create error message for all errors in an error structure
// Context is used to indicate whether the error structure has a parent 'dependencies', 'allOf', 'anyOf' or 'oneOf' error
static void CreateErrorMessages(const ValueType& errors, size_t depth = 0, const char* context = 0) {
    // Each member property contains one or more errors of a given type
    for (ValueType::ConstMemberIterator errorTypeItr = errors.MemberBegin(); errorTypeItr != errors.MemberEnd(); ++errorTypeItr) {
        const char* errorName = errorTypeItr->name.GetString();
        const ValueType& errorContent = errorTypeItr->value;
        if (errorContent.IsArray()) {
            // Member is an array where each item is an error - eg "type": [{"errorCode": ...}, {"errorCode": ...}]
            for (ValueType::ConstValueIterator contentItr = errorContent.Begin(); contentItr != errorContent.End(); ++contentItr) {
                HandleError(errorName, *contentItr, depth, context);
            }
        } else if (errorContent.IsObject()) {
            // Member is an object which is a single error - eg "type": {"errorCode": ... }
            HandleError(errorName, errorContent, depth, context);
        }
    }
}

bool validationJsonSchema(const char *test, const char *valid)
{
    rapidjson::Document test_doc;
    rapidjson::Document valid_doc;
    bool ret = false;
    valid_doc.Parse(valid);
    if(valid_doc.HasParseError()){
        std::cerr << "Schema validator document is not a valid JSON" << std::endl;
        std::cerr << "Error(offset " << static_cast<unsigned>(valid_doc.GetErrorOffset()) << "): "
                  << GetParseError_En(valid_doc.GetParseError()) << std::endl;
    }
    rapidjson::SchemaDocument schema_doc(valid_doc);
    rapidjson::SchemaValidator validator(schema_doc);
    rapidjson::Reader reader;
    rapidjson::StringStream test_stream(test);
    if(!reader.Parse(test_stream, validator) && reader.GetParseErrorCode() != rapidjson::kParseErrorTermination)
    {
        std::cerr << "Input is not a valid JSON" << std::endl;
        std::cerr << "Error(offset " << static_cast<unsigned>(reader.GetErrorOffset()) << "): "
                  << GetParseError_En(reader.GetParseErrorCode()) << std::endl;
    }
    if(validator.IsValid()){
        std::cout << "config json is valid" << std::endl;
        ret = true;
    }
    else {
        printf("Input JSON is invalid.\n");
        StringBuffer sb;
        validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
        std::cerr << "Invalid schema: " << sb.GetString() << std::endl;
        std::cerr << "Invalid keyword: " << validator.GetInvalidSchemaKeyword() << std::endl;
        std::cerr << "Invalid code: " << validator.GetInvalidSchemaCode() << std::endl;
        std::cerr << "Invalid message: " << GetValidateError_En(validator.GetInvalidSchemaCode()) << std::endl;
        sb.Clear();
        validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
        std::cerr << "Invalid document: " << sb.GetString() << std::endl;
        // Detailed violation report is available as a JSON value
        sb.Clear();
        PrettyWriter<StringBuffer> w(sb);
        validator.GetError().Accept(w);
        std::cerr << "Error report:\n" << sb.GetString() << std::endl;
        CreateErrorMessages(validator.GetError());
        return EXIT_FAILURE;
    }
    return ret;
}

// convert json to string
std::string JsonToString(const rapidjson::Value& val)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    val.Accept(writer);
    return buffer.GetString();
}

} // namespace dxapp